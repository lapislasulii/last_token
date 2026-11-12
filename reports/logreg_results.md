# Logistic regression: two-day validation

Fixed C=1.0; no hyperparameter search. Scaling and class encoding fitted inside each training fold.
Table excluded. Each day is predicted using only labels from the other day. No machine pool or manual-box features.

| Method | 2026-10-25 | 2026-10-26 | mean | worst |
|---|---:|---:|---:|---:|
| baseline | 0.860737 | 0.764807 | 0.812772 | 0.764807 |
| confidence | 0.860737 | 0.764807 | 0.812772 | 0.764807 |
| class | 0.810035 | 0.738848 | 0.774441 | 0.738848 |
| geometry | 0.808940 | 0.749006 | 0.778973 | 0.749006 |

confidence: confidence only; class: + one-hot class; geometry: + normalized size, area, center, bottom and pixel aspect ratio.
Mean is mean daily AP, not pooled AP. These small two-day estimates are not Kaggle scores.
OOF: oof/logreg_*.csv. Models: models/logreg_*.joblib. Test predictions: submissions/logreg_*.csv.
Final models refitted on all non-table train rows; each submission is validated against template IDs.
Environment: sklearn 1.9.1, pandas 3.1.0, numpy 2.5.4.
Train: 198 boxes. Test: 362 boxes.
