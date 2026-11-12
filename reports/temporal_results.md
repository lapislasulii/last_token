# Temporal experiments

Seven fixed configurations, each also evaluated with a fixed 50/50 baseline blend (14 candidates).
Two UTC-day folds; table excluded; scaling/encoding fit only on training fold.
Only supplied stream/context predictions used; no human boxes, pool, test labels or Kaggle feedback.
Neighbors across UTC-day boundaries excluded. Missing neighbors explicitly encoded in full variants.
CORE: maximum/minimum neighbor IoU, neighbor confidence difference, same-frame same-class overlap.
FULL: CORE + geometry + missing flags, actual time gaps, area/center changes and same-class count.

| Method | 2026-10-25 | 2026-10-26 | mean | worst | improves_both |
|---|---|---|---|---|---|
| baseline | 0.860737 | 0.764807 | 0.812772 | 0.764807 | False |
| temporal_core_c01 | 0.887502 | 0.746098 | 0.816800 | 0.746098 | False |
| temporal_core_c01_blend | 0.891329 | 0.774823 | 0.833076 | 0.774823 | True |
| temporal_core_c1 | 0.893799 | 0.766122 | 0.829961 | 0.766122 | True |
| temporal_core_c1_blend | 0.892035 | 0.788315 | 0.840175 | 0.788315 | True |
| temporal_class_c01 | 0.856341 | 0.741498 | 0.798920 | 0.741498 | False |
| temporal_class_c01_blend | 0.870090 | 0.772569 | 0.821330 | 0.772569 | True |
| temporal_class_c1 | 0.850639 | 0.742139 | 0.796389 | 0.742139 | False |
| temporal_class_c1_blend | 0.868452 | 0.774133 | 0.821293 | 0.774133 | True |
| temporal_full_c01 | 0.827758 | 0.771706 | 0.799732 | 0.771706 | False |
| temporal_full_c01_blend | 0.847332 | 0.773475 | 0.810403 | 0.773475 | False |
| temporal_full_c1 | 0.835309 | 0.785680 | 0.810495 | 0.785680 | False |
| temporal_full_c1_blend | 0.846256 | 0.772832 | 0.809544 | 0.772832 | False |
| temporal_trees | 0.813992 | 0.708799 | 0.761395 | 0.708799 | False |
| temporal_trees_blend | 0.855575 | 0.754496 | 0.805035 | 0.754496 | False |

Mean is mean daily AP. Selecting among candidates on these two days introduces selection bias.
Gains are exploratory, not independent validation or Kaggle scores. Prior logistic experiments also used these days.
Inspect per-day performance, not only the mean. Final published estimators are refitted on all 198 labeled non-table boxes.
