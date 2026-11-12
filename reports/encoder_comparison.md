# Сравнение размеров DINOv2

Локальное извлечение замороженных признаков на RTX 5070 Ti Laptop GPU. Encoder не дообучался. Одинаковые crop 1.5x с красной рамкой, 224×224, PCA16 и residual-голова; два UTC-day fold, все обучаемые трансформы внутри fold. Штрафы 1 и 10. Test используется только для готовых предсказаний, не обучения.

| Модель | AP 25 октября | AP 26 октября | Среднее |
|---|---:|---:|---:|
| marked_context_l1 | 0.863048 | 0.781325 | 0.822187 |
| marked_context_l10 | 0.866816 | 0.772977 | 0.819896 |
| marked_context_base_l1 | 0.863365 | 0.768754 | 0.816059 |
| marked_context_base_l10 | 0.870639 | 0.766642 | 0.818641 |
| marked_context_large_l1 | 0.872577 | 0.776306 | 0.824441 |
| marked_context_large_l10 | 0.873023 | 0.767509 | 0.820266 |

Large lambda=1 имеет наибольший средний AP, но уступает small lambda=1 на втором дне. Это небольшой исследовательский прирост, не независимое подтверждение и не Kaggle-результат. Текущая лучшая отправленная small-модель не заменена. Новых отправок не делали.

Публичные источники весов: https://huggingface.co/facebook/dinov2-base и https://huggingface.co/facebook/dinov2-large. Точные revision сохранены в outputs/marked_context_{base,large}/encoder.json и моделях.

Запуск: `python tools/train_marked_context.py --encoder base --batch-size 8` и `python tools/train_marked_context.py --encoder large --batch-size 4`. Small запускается без --encoder. Для каждого размера сохранены признаки train/test, OOF двух штрафов, финальная голова lambda=1 и тестовый CSV. CSV проверен по ID и диапазону, предсказания сверены после загрузки артефакта.

Следующий осмысленный тест — encoder другой архитектуры/предобучения или признаки внутри/снаружи бокса, а не дальнейшее наращивание DINOv2.
