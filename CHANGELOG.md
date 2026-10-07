# Changelog

## Unreleased

- Seed stability for all three models (seeds 42 to 46): logistic regression varies least (macro-F1 SD 0.0001 lncRNA), the MLP most (0.009); the lncRNA advantage holds for every seed with logistic regression and XGBoost but not the MLP.
- Nextflow seed-stability pipeline (`main.nf`, `modules/benchmark.nf`, `nextflow.config`, `scripts/18_seed_stability.py`): each gene universe x model x seed is a cached, resource-limited task; `-profile test` runs on synthetic data (`scripts/make_toy_data.py`, `configs/toy.yaml`) and in CI. Logistic regression over seeds 42 to 46: lncRNA macro-F1 SD 0.0001, protein-coding 0.0012, lncRNA ahead for every seed.
- `02_benchmark.py` takes `--config`, `--seed`, `--processed` and `--results` (`lncpan.config.with_overrides`).
- Fix: each logistic regression fit left joblib's worker pool idle for its 300 s timeout, and the isolated process could not exit until it expired; `run_isolated` now shuts the pool down. A toy benchmark went from about 10 minutes to under one.
- `Dataset.split` raises on a label outside the training classes instead of mapping it silently to a neighbouring class index (`np.searchsorted` returns a position for any value); tested in `tests/test_io.py`.
- README: the test set is not used for tuning, but it is not used only once either; the robustness analyses rescore it, and the sparse panel size is read off it.
- Fix: AJCC stage parsing dropped every sub-staged tumour ("Stage IIIA" and so on), leaving 528 of 1,868 test tumours with a stage instead of 1,226. Subgroup table, figure and model card regenerated; stage III remains the hardest (0.931), stage IV is 0.949, not 0.925. `parse_stage` is now tested.
- README: MET500 comparison now quotes the coding model's own test accuracy (96%, top-3 99.5%), so the external cohort costs about a third of its accuracy, not a quarter.
- Immune and stromal admixture (`scripts/17_microenvironment.py`, `make microenv`): test-set errors are not enriched in stroma- or immune-rich tumours and confidence barely depends on either; 17 of 330 top markers track the micro-environment within their type, and one thymoma marker (*TRBV11-2*) is a T-cell receptor segment typed lincRNA in GENCODE v23.

## 0.5.0 (2026-09-29)

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
