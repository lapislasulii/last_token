# Задание: ансамбль вторых детекторов (нужна видеокарта)

Исполнитель: Семён. Ветка `feat/redetect-ensemble` — это `codex/redetect-strong` плюс один коммит с поддержкой ансамбля в `tools/redetect.py`. Шаг 1 — около 70 минут, шаг 2 — около 80 минут, шаг 3 — по желанию на ночь.

## Зачем

Сильный детектор с вето дал Public **0.82471** против 0.70514 у одного вето. Локально прирост был всего +0.025, на Public — +0.12: дневные модели учатся на 42 кадрах, итоговая на 84, и качество детектора растёт с данными быстрее, чем показывает проверка по дням. Значит, рычаг сейчас — качество и устойчивость второго детектора.

Где одна модель шумит. Доля исходных боксов, у которых IoU с новым детектором лежит у самого порога (0.5–0.7):

| Класс | train, дневные модели | test, итоговая модель |
|---|---:|---:|
| `cup` | 39% | 33% |
| `nude_hands` | 17% | 24% |
| `person` | 25% | 20% |
| `tray` | 28% | 12% |

Для таких боксов решение «есть пара / нет пары» зависит от случайности обучения. Ансамбль заменяет жёсткое 0 или 1 на долю моделей, которые не согласны с боксом.

## Что изменилось в `tools/redetect.py`

- `--seed N` — seed обучения (раньше был зашит 0).
- `--member NAME` для `train` и `predict` — имя участника ансамбля. Запуск и предсказания получают суффикс: `runs/all_s1`, `pred_all_s1.csv`. Участник `base` — уже обученные модели без суффикса, их переобучать не надо.
- `--members base s1 ...` для `eval` и `submit` — собрать ансамбль. `D` становится долей участников без пары с IoU ≥ 0.6, формула та же: `(1 − confidence + D) / 2`.
- С одним участником всё работает как раньше; проверено на твоих файлах: `eval --tag strong` даёт прежние 0.9147 / 0.8455.

Датасеты уже собраны у тебя в `outputs/redetect_strong/yolo/`, `build` повторять не нужно.

## Шаг 1. Ансамбль для test (около 70 минут) — сделать сегодня

Две дополнительные модели на обоих днях, рецепт тот же, меняется только seed:

```
python tools/redetect.py train --tag strong --member s1 --seed 1 --weights weights/yolo11l.pt --epochs 250 --batch 4 --device 0 --sets all
python tools/redetect.py predict --tag strong --member s1 --device 0 --sets all
python tools/redetect.py train --tag strong --member s2 --seed 2 --weights weights/yolo11l.pt --epochs 250 --batch 4 --device 0 --sets all
python tools/redetect.py predict --tag strong --member s2 --device 0 --sets all
python tools/redetect.py submit --tag strong --members base s1 s2
python tools/blend_veto.py test submissions/redetect_strong_ens_veto.csv
```

Результат: `submissions/redetect_strong_ens_veto_veto.csv`. Прислать файл и закоммитить `outputs/redetect_strong/pred_all_s1.csv`, `pred_all_s2.csv`, `test_features_ens.csv`.

Отправку в Kaggle согласовать: сегодня попытки ещё есть.

## Шаг 2. Проверка ансамбля по дням (около 80 минут)

Нужна для отчёта и для выбора финальных отправок: по одному Public выбирать нельзя.

```
python tools/redetect.py train --tag strong --member s1 --seed 1 --weights weights/yolo11l.pt --epochs 250 --batch 4 --device 0
python tools/redetect.py predict --tag strong --member s1 --device 0
python tools/redetect.py train --tag strong --member s2 --seed 2 --weights weights/yolo11l.pt --epochs 250 --batch 4 --device 0
python tools/redetect.py predict --tag strong --member s2 --device 0
python tools/redetect.py eval --tag strong --members base s1 s2
python tools/blend_veto.py train oof/redetect_strong_ens_veto.csv
```

`eval` печатает AP каждого участника и ансамбля. Закоммитить `pred_day_*_s1.csv`, `pred_day_*_s2.csv`, `oof_features_ens.csv`.

**Критерий, заданный заранее:** среднее AP ансамбля не ниже среднего по трём участникам, и ни один день не хуже лучшего участника больше чем на 0.01. Если выполнен — ансамбль идёт первым кандидатом в финальные отправки. Если нет — остаётся одиночная модель (0.82471).

## Шаг 3. По желанию на ночь: участник другого размера

`yolo11x` ошибается иначе, чем `yolo11l`, поэтому в ансамбле полезнее ещё одного seed. Скорее всего, потребуется `--batch 2`; время обучения нужно измерить.

```
python tools/redetect.py train --tag strong --member x --seed 0 --weights weights/yolo11x.pt --epochs 250 --batch 2 --device 0 --sets all
python tools/redetect.py predict --tag strong --member x --device 0 --sets all
python tools/redetect.py submit --tag strong --members base s1 s2 x
```

Для проверки по дням — те же команды без `--sets all`, затем `eval --members base s1 s2 x`.

## Правила

- Рецепт не менять: те же 250 эпох, размер 1280, 12 классов. Меняется только seed (и размер модели в шаге 3).
- Эпоху и состав ансамбля по отложенному дню не подбирать: участники берутся все, какие обучены.
- Веса и картинки в Git не класть; таблицы предсказаний — можно, как в прошлый раз.
