# Прогнозы ансамбля на test

В outputs/redetect_strong файлы pred_all_flip.csv, pred_all_s1_flip.csv и pred_all_s2_flip.csv содержат рамки трёх YOLO на отражённых кадрах test. Координаты уже переведены обратно в исходный кадр. Файлы pred_all.csv, pred_all_s1.csv и pred_all_s2.csv — соответствующие обычные прогоны. В каждой строке находится одна детекция модели. Для просмотра итогового score исходного бокса используйте CSV из submissions.

RF-DETR 1120 обучен на четырёх классах. pred_all_rfdetr1120.csv содержит его рамки на test. В имени pred_day_*_rfdetr1120.csv указан день обучения, прогноз сделан на другом дне. Основной ансамбль усредняет долю возражений YOLO и голос RF-DETR поровну, сохраняя прежний вес позы у людей вне окна. Локальный AP: 0,94658 и 0,92671. Отправка Kaggle № 58340736 получила Public AP 0,90664. У опорного ансамбля Public AP 0,92465.

Просмотр из корня репозитория с локальным датасетом:

    python tools/view_frame.py --split test --test-scores submissions/redetect_rfdetr1120_family_pvote_nowin_legs_veto.csv
