# CLIP ViT-L/14: другой encoder

Источник весов: https://huggingface.co/openai/clip-vit-large-patch14 . Revision 32bd64288804d66eefd0ccbe215aa642df71cc41. Замороженный CLIPVisionModel, post-layernorm CLS (1024 признака), без текстовой ветки и projection. Нормализация из опубликованного image processor CLIP; изображение то же: crop1.5x, pad224, рамка2px. PCA16 и остаточная линейная голова обучены внутри каждого day fold. Два штрафа, lambda=1 и 10.

| Метод | AP25 | AP26 | Среднее |
|---|---:|---:|---:|
| DINOv2-small lambda1 | 0.863048 | 0.781325 | 0.822187 |
| DINOv2-large lambda1 | 0.872577 | 0.776306 | 0.824441 |
| CLIP lambda1 | 0.867744 | 0.768759 | 0.818251 |
| CLIP lambda10 | 0.870714 | 0.768831 | 0.819773 |
| Fixed 50/50 small+CLIP lambda1 | 0.863316 | 0.772448 | 0.817882 |

CLIP не показал улучшения самостоятельной модели в среднем. Это вывод для конкретной frozen CLS/PCA16 схемы, не для всех применений CLIP. Сравнение уже использует многократно проверенные два дня; независимого подтверждения нет. Kaggle не отправляли.

Запуск: `python tools/train_marked_context.py --encoder clip --batch-size 4`. Признаки, revision и метрики: outputs/marked_context_clip/. Артефакт lambda1 и проверенный CSV сохранены в models/ и submissions/. Все 362 ID совпали с шаблоном, значения конечны и в [0,1], восстановленная голова воспроизводит предсказания. CLIP text/projection веса при загрузке vision-only не используются: сообщения UNEXPECTED об этих ключах ожидаемы.
