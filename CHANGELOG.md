# Changelog

## Unreleased

- Sparse lncRNA panel by L1 selection and L2 refit (`scripts/16_sparse_panel.py`): a few hundred genes are needed, not a few dozen; the 376-gene panel is listed.
- Bootstrap stability of the marker lncRNAs (`scripts/14_marker_stability.py`) and a direct measurement of the hospital-site signal within cancer types (`scripts/15_site_signal.py`).
- Calibration under shift and results by biopsy site on MET500 (`lncpan.calibration`, `scripts/13_met500_calibration.py`): temperature scaling fitted on TCGA barely helps on MET500; liver and lung biopsies are often called the host organ's cancer.
- Randomised APS and regularised adaptive prediction sets (RAPS) in `lncpan.conformal`, with the RAPS constants tuned on a split of the calibration set; `lncpan predict --score raps`. The deterministic APS score gave 14-type sets and empty sets to the most confident tumours; the randomised APS score covers 94% of the external metastases with one global threshold.

## 0.4.0 (2026-09-29)

- External validation on the MET500 metastatic cohort (`scripts/12_met500.py`): protein-coding model 0.66 top-1 / 0.82 top-3; the public matrix has no lncRNAs, so the lncRNA model abstains.
- `lncpan predict` command: probabilities and conformal sets for new samples from the saved model and thresholds.
- Dockerfile, `make docker`, pre-commit configuration.

## 0.3.0 (2026-09-28)

- Held-out hospital site cross-validation (`scripts/11_site_holdout.py`): about four points of macro-F1 are lost when every tissue source site is unseen in training.
- Citation file.

## 0.2.0 (2026-09-28)

- Conformal prediction sets with marginal and class-conditional guarantees (`lncpan.conformal`, `scripts/08_conformal.py`), evaluated on the test split and on the external melanoma metastases.
- Subgroup analysis by sex, age, AJCC stage and class rarity, with bootstrap confidence intervals (`scripts/09_subgroups.py`).
- Learning curve for the lncRNA and protein-coding models (`scripts/10_learning_curve.py`).
- Model card (`MODEL_CARD.md`).
- README shortened; results unchanged.

## 0.1.0 (2026-09-24)

- Benchmark of logistic regression, XGBoost and a PyTorch MLP on lncRNA and protein-coding genes, 9,337 TCGA tumours, 33 cancer types.
- Feature-budget analysis, SHAP interpretation, external melanoma metastases, sex-chromosome ablation.
