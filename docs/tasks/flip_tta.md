# Задание: отражение кадров (TTA) для ансамбля redetect

Исполнитель: Семён (GPU). Ветка `codex/redetect-strong`. Время: ~10–15 минут, без обучения.

## Идея

Каждая модель ансамбля дополнительно прогоняется по кадрам, отражённым по горизонтали, а боксы отражаются обратно.
Отражённый прогон — ещё один голосующий участник в D: `s1_flip` — те же веса s1, отражённые кадры.
Код: `predict --flip` пишет `pred_<набор>[_участник]_flip.csv`, в `--members` такие участники пишутся с суффиксом `_flip`.

## Что уже проверено локально

- Обратное отражение правильное: на слабой YOLO11s уверенные боксы обычного и отражённого прогона совпадают,
  медианный IoU 0.76–0.82.
- **Слабая YOLO11s + вето: base 0.8673, base + base_flip 0.8559 — отражение ухудшило.** Сильные модели могут вести
  себя иначе, но ожидание скромное.

## Шаги

Все команды из корня репо. Правило «один к одному» уже в коде, поэтому сравнение идёт с новыми числами.

### 1. Проверка по дням

```bash
python tools/redetect.py predict --tag strong --member base --flip --device 0
python tools/redetect.py predict --tag strong --member s1   --flip --device 0
python tools/redetect.py predict --tag strong --member s2   --flip --device 0
python tools/redetect.py eval --tag strong --members base s1 s2
python tools/redetect.py eval --tag strong --members base s1 s2 base_flip s1_flip s2_flip
python tools/redetect.py eval --tag strong --members base base_flip
```

### 2. Критерий (зафиксирован до запуска)

Сравниваем строки `ансамбль ... + вето` двух первых `eval`.

- **Успех:** шесть участников дают среднее не меньше, чем три участника + 0.003, и ни один день не хуже больше чем на 0.005.
- Иначе бросаем. Порогов, весов и состава не перебирать.

### 3. Если успех — test

```bash
python tools/redetect.py predict --tag strong --member base --flip --sets all --device 0
python tools/redetect.py predict --tag strong --member s1   --flip --sets all --device 0
python tools/redetect.py predict --tag strong --member s2   --flip --sets all --device 0
python tools/redetect.py submit --tag strong --members base s1 s2 base_flip s1_flip s2_flip
python tools/blend_veto.py test submissions/redetect_strong_ens-base-s1-s2-base_flip-s1_flip-s2_flip_veto.csv
```

Сравнить с текущим лучшим (Public 0.90211) через аудитор Маши: `--vs submissions/redetect_strong_ens-base-s1-s2_1to1_veto_veto.csv`.

## Что вернуть

Вывод трёх `eval` — в `reports/flip_tta.md`. Файлы `pred_*_flip.csv` можно запушить с `git add -f`, как `pred_all_s1.csv`:
тогда test-файл соберётся на любой машине.
