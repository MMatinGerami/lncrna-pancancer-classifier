# Changelog

## 0.2.0 (2026-09-28)

- Conformal prediction sets with marginal and class-conditional guarantees (`lncpan.conformal`, `scripts/08_conformal.py`), evaluated on the test split and on the external melanoma metastases.
- Subgroup analysis by sex, age, AJCC stage and class rarity, with bootstrap confidence intervals (`scripts/09_subgroups.py`).
- Learning curve for the lncRNA and protein-coding models (`scripts/10_learning_curve.py`).
- Model card (`MODEL_CARD.md`).
- README shortened; results unchanged.

## 0.1.0 (2026-09-24)

- Benchmark of logistic regression, XGBoost and a PyTorch MLP on lncRNA and protein-coding genes, 9,337 TCGA tumours, 33 cancer types.
- Feature-budget analysis, SHAP interpretation, external melanoma metastases, sex-chromosome ablation.
