"""Единый evaluator для Issue #1: проверка CSV id,score и сравнение AP по UTC-дням.

Запуск из корня репозитория:
  python tools/evaluate_predictions.py info
  python tools/evaluate_predictions.py baseline --mode oof
  python tools/evaluate_predictions.py score my_oof.csv --mode oof --name my_method
  python tools/evaluate_predictions.py score day.csv --mode day --day 2026-10-25 --name my_day_method
  python tools/evaluate_predictions.py selftest

Глобальные параметры (например, --data) передаются до команды:
  python tools/evaluate_predictions.py --data path/to/data info

Входной CSV: id,score; дополнительные столбцы разрешены. Сопоставление только по id.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "rascar-hack-case-1"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

# Точные контрольные значения из Issue #1 / плана команды.
REFERENCE = {
    "baseline_1_minus_conf": {
        "2026-10-25": 0.860737,
        "2026-10-26": 0.764807,
        "mean": 0.812772,
    },
    "temporal_core_c1_blend": {
        "2026-10-25": 0.892035,
        "2026-10-26": 0.788315,
        "mean": 0.840175,
    },
}
REFERENCE_TOL = 1e-6


class EvaluationError(ValueError):
    """Некорректные данные или предсказания."""


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Загружает train.csv, присоединяет UTC-день по frame_id, исключает table/запрещённые классы."""
    policy_path = data_dir / "scoring_policy.json"
    train_path = data_dir / "train.csv"
    frames_path = data_dir / "train_frames.csv"
    for path in (policy_path, train_path, frames_path):
        if not path.exists():
            raise EvaluationError(f"Не найден файл: {path}")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    train = pd.read_csv(train_path, dtype={"id": str, "frame_id": str})
    required = {"id", "frame_id", "class_name", "confidence", "error"}
    missing_cols = required - set(train.columns)
    if missing_cols:
        raise EvaluationError(f"train.csv: отсутствуют колонки {sorted(missing_cols)}")
    if train["id"].isna().any() or train["id"].duplicated().any():
        raise EvaluationError("train.csv: id пустые или повторяются")

    frames = pd.read_csv(frames_path, dtype={"frame_id": str})
    if not {"frame_id", "timestamp"} <= set(frames.columns):
        raise EvaluationError("train_frames.csv: нужны колонки frame_id,timestamp")
    if frames["frame_id"].duplicated().any():
        raise EvaluationError("train_frames.csv: повторяющиеся frame_id")
    timestamps = pd.to_datetime(frames["timestamp"], utc=True, errors="coerce", format="mixed")
    if timestamps.isna().any():
        raise EvaluationError("train_frames.csv: есть некорректные timestamp")
    frames = frames.assign(day=timestamps.dt.strftime("%Y-%m-%d"))
    day_by_frame = frames.set_index("frame_id")["day"]
    train["day"] = train["frame_id"].map(day_by_frame)
    if train["day"].isna().any():
        count = int(train["day"].isna().sum())
        raise EvaluationError(f"{count} строк train.csv не имеют соответствующего frame_id в train_frames.csv")

    train["class_name"] = train["class_name"].replace(policy.get("aliases", {}))
    excluded = set(policy.get("test_excluded_classes", [])) | {"table"}
    train = train.loc[~train["class_name"].isin(excluded)].copy()
    train["error"] = pd.to_numeric(train["error"], errors="coerce")
    if train["error"].isna().any() or not train["error"].isin([0, 1]).all():
        raise EvaluationError("train.csv: error должен содержать только 0/1")
    train["error"] = train["error"].astype(int)
    train["confidence"] = pd.to_numeric(train["confidence"], errors="coerce")
    if train["confidence"].isna().any() or not np.isfinite(train["confidence"]).all():
        raise EvaluationError("train.csv: confidence содержит NaN/inf/нечисловые значения")
    return train.reset_index(drop=True)


def validate_prediction_file(path: Path, expected_ids: pd.Series) -> tuple[pd.Series, list[str]]:
    """Проверяет полный набор ID и score. Возвращает score, индексированный ожидаемыми ID."""
    try:
        pred = pd.read_csv(path, dtype={"id": str})
    except Exception as exc:
        raise EvaluationError(f"Не удалось прочитать CSV {path}: {exc}") from exc
    if not {"id", "score"} <= set(pred.columns):
        raise EvaluationError(f"{path.name}: нужны колонки id,score; получены {list(pred.columns)}")
    if pred["id"].isna().any() or pred["score"].isna().any():
        raise EvaluationError("CSV предсказаний содержит пустые id или score")

    ids = pred["id"].astype(str)
    if ids.duplicated().any():
        examples = ids[ids.duplicated()].head(5).tolist()
        raise EvaluationError(f"Повторяющиеся id: {examples}")

    expected = pd.Index(expected_ids.astype(str))
    actual = pd.Index(ids)
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        bits = []
        if len(missing):
            bits.append(f"потеряно ID: {len(missing)}, например {missing[:5].tolist()}")
        if len(extra):
            bits.append(f"неизвестные ID: {len(extra)}, например {extra[:5].tolist()}")
        raise EvaluationError("; ".join(bits))

    score = pd.to_numeric(pred["score"], errors="coerce")
    if score.isna().any():
        raise EvaluationError("score содержит нечисловые или NaN значения")
    values = score.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise EvaluationError("score содержит +inf или -inf")
    if ((values < 0) | (values > 1)).any():
        bad = values[(values < 0) | (values > 1)]
        raise EvaluationError(f"score вне диапазона [0,1]: min={bad.min()}, max={bad.max()}")

    aligned = pd.Series(values, index=ids).reindex(expected)
    warnings = []
    if aligned.nunique() == 1:
        warnings.append("все score одинаковы: ранжирования нет")
    elif aligned.nunique() < len(aligned):
        warnings.append(f"есть совпадающие score: {aligned.nunique()} уникальных значений из {len(aligned)}")
    return aligned, warnings


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    """Официальный sklearn AP для непустой группы; None только для пустой."""
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return None
    return float(average_precision_score(y, score))


def evaluate(df: pd.DataFrame, scores: pd.Series) -> dict:
    scores = scores.reindex(df["id"].astype(str))
    if scores.isna().any():
        raise EvaluationError("После выравнивания по id остались отсутствующие score")

    per_day = []
    for day, group in df.groupby("day", sort=True):
        y = group["error"].to_numpy(dtype=int)
        s = scores.loc[group["id"].astype(str)].to_numpy(dtype=float)
        baseline = 1.0 - group["confidence"].to_numpy(dtype=float)
        model_ap = average_precision(y, s)
        baseline_ap = average_precision(y, baseline)
        per_day.append({
            "day": day,
            "n_boxes": int(len(group)),
            "n_errors": int(y.sum()),
            "ap": model_ap,
            "baseline_ap": baseline_ap,
            "delta_vs_baseline": None if model_ap is None or baseline_ap is None else model_ap - baseline_ap,
        })

    valid_ap = [row["ap"] for row in per_day if row["ap"] is not None]
    valid_base = [row["baseline_ap"] for row in per_day if row["baseline_ap"] is not None]
    mean_ap = float(np.mean(valid_ap)) if valid_ap else None
    mean_baseline = float(np.mean(valid_base)) if valid_base else None
    deltas = [row["delta_vs_baseline"] for row in per_day if row["delta_vs_baseline"] is not None]

    per_class = []
    for class_name, group in df.assign(_score=scores.to_numpy()).groupby("class_name", sort=True):
        y = group["error"].to_numpy(dtype=int)
        ap_value = average_precision(y, group["_score"].to_numpy(dtype=float))
        baseline_ap = average_precision(y, 1.0 - group["confidence"].to_numpy(dtype=float))
        per_class.append({
            "class_name": str(class_name),
            "n_boxes": int(len(group)),
            "n_errors": int(y.sum()),
            "ap": ap_value,
            "ap_status": "ok" if 0 < y.sum() < len(y) else "неинформативен: нет положительных или нет отрицательных примеров",
            "baseline_ap": baseline_ap,
        })

    return {
        "per_day": per_day,
        "summary": {
            "mean_ap": mean_ap,
            "worst_day_ap": min(valid_ap) if valid_ap else None,
            "mean_baseline_ap": mean_baseline,
            "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else None,
        },
        "per_class": per_class,
    }


def select_rows(df: pd.DataFrame, mode: str, day: str | None) -> pd.DataFrame:
    if mode == "oof":
        if day is not None:
            raise EvaluationError("--day используется только с --mode day")
        return df.copy()
    if mode == "day":
        if not day:
            raise EvaluationError("Для --mode day необходимо указать --day YYYY-MM-DD")
        selected = df.loc[df["day"] == day].copy()
        if selected.empty:
            raise EvaluationError(f"В датасете нет строк за день {day}")
        return selected.reset_index(drop=True)
    raise EvaluationError(f"Неизвестный режим: {mode}")


def compare_reference(name: str, result: dict) -> dict:
    reference = REFERENCE.get(name)
    if reference is None:
        return {"checked": False, "message": "для этого метода нет зафиксированного эталона"}
    actual_by_day = {row["day"]: row["ap"] for row in result["per_day"]}
    actual_mean = result["summary"]["mean_ap"]
    checks = {}
    for day, expected in reference.items():
        actual = actual_mean if day == "mean" else actual_by_day.get(day)
        checks[day] = {
            "expected": expected,
            "actual": actual,
            "ok": actual is not None and abs(actual - expected) <= REFERENCE_TOL,
        }
    return {"checked": True, "tolerance": REFERENCE_TOL, "checks": checks,
            "ok": all(item["ok"] for item in checks.values())}


def write_outputs(name: str, mode: str, day: str | None, result: dict, reference: dict) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "method": name,
        "mode": mode,
        "day": day,
        "created_at_utc": stamp,
        **result,
        "reference_check": reference,
    }
    json_path = OUTPUT_DIR / f"{safe_name}_{mode}_{day or 'all'}_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = OUTPUT_DIR / "experiments_summary.csv"
    row = {
        "method": name, "mode": mode, "day": day or "",
        "mean_ap": result["summary"]["mean_ap"],
        "worst_day_ap": result["summary"]["worst_day_ap"],
        "mean_baseline_ap": result["summary"]["mean_baseline_ap"],
        "mean_delta_vs_baseline": result["summary"]["mean_delta_vs_baseline"],
        "reference_ok": reference.get("ok") if reference.get("checked") else None,
        "created_at_utc": stamp,
    }
    new = pd.DataFrame([row])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[~((old["method"] == name) & (old["mode"] == mode) & (old["day"].fillna("") == (day or "")))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(summary_path, index=False)
    return json_path, summary_path


def print_result(name: str, result: dict, reference: dict) -> None:
    print(f"\n=== {name} ===")
    print("AP по дням:")
    for row in result["per_day"]:
        ap_text = "неинформативен" if row["ap"] is None else f"{row['ap']:.6f}"
        base_text = "неинформативен" if row["baseline_ap"] is None else f"{row['baseline_ap']:.6f}"
        delta_text = "—" if row["delta_vs_baseline"] is None else f"{row['delta_vs_baseline']:+.6f}"
        print(f"  {row['day']}: AP={ap_text}; baseline={base_text}; delta={delta_text}; "
              f"боксов={row['n_boxes']}; ошибок={row['n_errors']}")
    sm = result["summary"]
    fmt = lambda x: "неинформативен" if x is None else f"{x:.6f}"
    print(f"Средний AP: {fmt(sm['mean_ap'])}")
    print(f"Худший AP по дням: {fmt(sm['worst_day_ap'])}")
    print(f"Средний baseline AP: {fmt(sm['mean_baseline_ap'])}")
    print(f"Средняя разница с baseline: {fmt(sm['mean_delta_vs_baseline'])}")
    if reference.get("checked"):
        print(f"Эталонная сверка (допуск {REFERENCE_TOL}): {'OK' if reference['ok'] else 'НЕ ПРОЙДЕНА'}")
        for key, item in reference["checks"].items():
            print(f"  {key}: получено {item['actual']}; ожидалось {item['expected']}; "
                  f"{'OK' if item['ok'] else 'FAIL'}")
    print("Диагностика классов:")
    for row in result["per_class"]:
        ap_text = "неинформативен" if row["ap"] is None else f"{row['ap']:.6f}"
        print(f"  {row['class_name']}: боксов={row['n_boxes']}, ошибок={row['n_errors']}, "
              f"AP={ap_text} ({row['ap_status']})")


def run_evaluation(df: pd.DataFrame, scores: pd.Series, name: str, mode: str, day: str | None) -> int:
    result = evaluate(df, scores)
    reference = compare_reference(name, result) if mode == "oof" else {
        "checked": False, "message": "эталонная сверка выполняется только на полном OOF"
    }
    print_result(name, result, reference)
    json_path, csv_path = write_outputs(name, mode, day, result, reference)
    print(f"\nJSON-отчёт: {json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path}")
    print(f"Сводная таблица: {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}")
    return 0 if not reference.get("checked") or reference.get("ok") else 2


def command_info(data: Path) -> None:
    df = load_dataset(data)
    print(f"Оцениваемых боксов (без table и исключённых классов): {len(df)}")
    print(f"Ошибок: {int(df.error.sum())}")
    print(df.groupby("day").agg(frames=("frame_id", "nunique"), boxes=("id", "size"), errors=("error", "sum")))
    if len(df) != 198 or int(df.error.sum()) != 109:
        print("ВНИМАНИЕ: количество строк/ошибок отличается от контрольного описания Issue #1 (198/109).")


def command_baseline(data: Path, mode: str, day: str | None) -> int:
    df = select_rows(load_dataset(data), mode, day)
    scores = pd.Series(1.0 - df["confidence"].to_numpy(dtype=float), index=df["id"].astype(str))
    return run_evaluation(df, scores, "baseline_1_minus_conf", mode, day)


def command_score(data: Path, file: Path, mode: str, day: str | None, name: str) -> int:
    df = select_rows(load_dataset(data), mode, day)
    scores, warnings = validate_prediction_file(file, df["id"])
    for warning in warnings:
        print("ВНИМАНИЕ:", warning)
    return run_evaluation(df, scores, name, mode, day)


def selftest() -> None:
    """Проверяет порядок строк, ties и ошибки входного файла на синтетических данных."""
    import tempfile

    ids = pd.Series(["a", "b", "c", "d"])
    y = np.array([1, 0, 1, 0])
    scores = pd.Series([0.9, 0.8, 0.7, 0.1], index=ids)
    frame = pd.DataFrame({
        "id": ids, "day": ["2026-10-25"] * 4, "error": y,
        "class_name": ["cup"] * 4, "confidence": [0.2, 0.8, 0.3, 0.9],
    })
    original = evaluate(frame, scores)
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    shuffled_scores = scores.reindex(shuffled.id)
    permuted = evaluate(shuffled, shuffled_scores)
    assert np.isclose(original["summary"]["mean_ap"], permuted["summary"]["mean_ap"])
    print("OK: перестановка строк не меняет AP")

    tied = pd.Series([0.5, 0.5, 0.5, 0.5], index=ids)
    tied_result = evaluate(frame, tied)
    assert np.isclose(tied_result["summary"]["mean_ap"], y.mean())
    print("OK: tied scores обрабатываются")

    # A single-class day must remain in the official average, while class
    # diagnostics can still explain that there is no ranking discrimination.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert average_precision(np.array([]), np.array([])) is None
        for label in (0, 1):
            labels = np.full(4, label)
            expected = average_precision_score(labels, scores.to_numpy())
            assert average_precision(labels, scores.to_numpy()) == expected
            single = frame.assign(error=labels)
            single_result = evaluate(single, scores)
            assert single_result["per_day"][0]["ap"] == expected
            assert single_result["summary"]["mean_ap"] == expected
            assert single_result["per_class"][0]["ap"] == expected
            assert single_result["per_class"][0]["ap_status"].startswith("неинформативен")
            second = single.assign(id=single.id + "_next", day="2026-10-26")
            combined = pd.concat([frame, second], ignore_index=True)
            combined_scores = pd.concat([scores, pd.Series(scores.to_numpy(), index=second.id)])
            result = evaluate(combined, combined_scores)
            expected_mean = (original["summary"]["mean_ap"] + expected) / 2
            assert np.isclose(result["summary"]["mean_ap"], expected_mean)
            assert np.isclose(result["summary"]["mean_baseline_ap"],
                              (original["summary"]["mean_baseline_ap"] + expected) / 2)
            assert result["summary"]["worst_day_ap"] == min(original["summary"]["mean_ap"], expected)
            assert np.isclose(result["summary"]["mean_delta_vs_baseline"],
                              original["summary"]["mean_delta_vs_baseline"] / 2)
    print("OK: дни с одними 0 или 1 учитываются в AP и среднем; диагностический статус сохранён")

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "pred.csv"
        pd.DataFrame({"id": ids[::-1], "score": [0.1, 0.7, 0.8, 0.9]}).to_csv(p, index=False)
        aligned, _ = validate_prediction_file(p, ids)
        assert aligned.index.tolist() == ids.tolist()
        print("OK: сопоставление по id, а не по порядку строк")

        bad_cases = [
            (pd.DataFrame({"id": ["a", "a", "c", "d"], "score": [0.1, 0.2, 0.3, 0.4]}), "дубли"),
            (pd.DataFrame({"id": ["a", "b", "c"], "score": [0.1, 0.2, 0.3]}), "потерянный id"),
            (pd.DataFrame({"id": ["a", "b", "c", "x"], "score": [0.1, 0.2, 0.3, 0.4]}), "неизвестный id"),
            (pd.DataFrame({"id": ids, "score": [0.1, np.nan, 0.3, 0.4]}), "NaN"),
            (pd.DataFrame({"id": ids, "score": [0.1, np.inf, 0.3, 0.4]}), "inf"),
            (pd.DataFrame({"id": ids, "score": [0.1, 1.2, 0.3, 0.4]}), "диапазон"),
        ]
        for bad, label in bad_cases:
            bad.to_csv(p, index=False)
            try:
                validate_prediction_file(p, ids)
            except EvaluationError:
                print(f"OK: обнаружен некорректный ввод ({label})")
            else:
                raise AssertionError(f"Не обнаружена ошибка: {label}")
    print("SELFTEST ПРОЙДЕН")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="папка с датасетом")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info")

    baseline = sub.add_parser("baseline")
    baseline.add_argument("--mode", choices=["day", "oof"], default="oof")
    baseline.add_argument("--day")

    score = sub.add_parser("score")
    score.add_argument("file", type=Path)
    score.add_argument("--mode", choices=["day", "oof"], default="oof")
    score.add_argument("--day")
    score.add_argument("--name", required=True)

    sub.add_parser("selftest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "selftest":
            selftest()
            return 0
        if args.command == "info":
            command_info(args.data)
            return 0
        if args.command == "baseline":
            return command_baseline(args.data, args.mode, args.day)
        if args.command == "score":
            return command_score(args.data, args.file, args.mode, args.day, args.name)
    except EvaluationError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
