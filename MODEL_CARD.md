# Model card: lncRNA tissue-of-origin classifier

Research prototype. Not a medical device and not validated for clinical use.

## Model

- **Task:** predict the tissue of origin (33 TCGA cancer types) of a tumour from its lncRNA expression profile.
- **Model:** L2-regularised multinomial logistic regression (C = 0.1) on the 2,000 most variable autosomal lncRNAs, selected on the training data only, after standardisation. XGBoost and a PyTorch MLP were benchmarked and did not perform better (`results/tables/metrics_all.csv`).
- **Input:** log2(TPM + 1) expression of lncRNA genes (GENCODE v23 biotypes), from the UCSC Xena TCGA Toil recompute.
- **Output:** a probability for each of the 33 cancer types. With the conformal procedure in `scripts/08_conformal.py`, a set of candidate types with a stated error rate.

## Data

- **Source:** TCGA primary tumours, one sample per patient, cancer types with at least 30 samples.
- **Splits:** 7,469 training and 1,868 test tumours (stratified, 80/20, seed 42); patients never appear in both.
- **External set:** 364 melanoma metastases (TCGA sample type 06) from patients absent from training.
- **Sex chromosomes** are excluded so that the model cannot use patient sex to recognise sex-specific cancers; `scripts/07_ablation_sex_chromosomes.py` shows what changes when they are included.

## Performance (test split, 95% bootstrap CIs)

| Metric | Value |
|---|---|
| Macro-F1 | 0.946 (0.928 to 0.959) |
| Accuracy | 0.966 |
| Top-3 accuracy | 0.996 |
| Expected calibration error | 0.011 |
| Melanoma metastases assigned to skin | 93% (top-3: 97%) |

## Subgroup performance (test split, lncRNA model)

| Subgroup | n | Accuracy (95% CI) | ECE |
|---|---|---|---|
| Female | 918 | 0.969 (0.959 to 0.980) | 0.015 |
| Male | 950 | 0.963 (0.951 to 0.975) | 0.010 |
| Age under 50 | 480 | 0.985 (0.973 to 0.996) | 0.012 |
| Age 50 to 64 | 725 | 0.957 (0.942 to 0.971) | 0.015 |
| Age 65 and over | 653 | 0.962 (0.948 to 0.975) | 0.016 |
| AJCC stage I | 219 | 0.973 (0.950 to 0.991) | 0.017 |
| AJCC stage II | 121 | 0.983 (0.959 to 1.000) | 0.014 |
| AJCC stage III | 108 | 0.935 (0.889 to 0.981) | 0.034 |
| AJCC stage IV | 80 | 0.925 (0.862 to 0.975) | 0.047 |
| Class with 300+ training tumours | 1,182 | 0.975 (0.965 to 0.984) | 0.009 |
| Class with 100 to 299 training tumours | 536 | 0.968 (0.951 to 0.981) | 0.017 |
| Class with fewer than 100 training tumours | 150 | 0.887 (0.840 to 0.933) | 0.073 |

The clear weakness is rare cancer types: accuracy drops by about nine points and the model is over-confident on them. Late-stage tumours are somewhat harder, with wide intervals. No sex difference is detectable.

## Uncertainty

Split conformal prediction (`scripts/08_conformal.py`, 15% of the training split held out for calibration):

| Setting | Coverage (test) | Coverage (metastases) | Mean set size | Empty sets |
|---|---|---|---|---|
| LAC, α = 0.10, one global threshold | 0.896 | 0.761 | 0.9 | 9.3% |
| LAC, α = 0.10, one threshold per class | 0.918 | 0.945 | 4.9 | 0% |
| LAC, α = 0.05, one global threshold | 0.952 | 0.901 | 1.0 | 2.3% |

An empty set is an abstention; among non-empty sets at α = 0.10, 98.8% contain the true class. The global threshold loses its guarantee on the metastases (a distribution shift), while per-class thresholds keep it at the cost of larger sets.

## Limitations

- TCGA only: one consortium, bulk RNA-seq, mostly primary tumours from North American centres. Performance on other platforms, formalin-fixed samples or other populations is unknown.
- Rare classes (fewer than 100 training tumours) are less accurate and less well calibrated.
- Cancers of the same organ (colon vs rectum, oesophagus vs stomach) are not reliably separated; this reflects biology as much as the model.
- Coverage guarantees assume test data are exchangeable with the calibration data, which the metastases result shows does not hold under shift.
- Clinical metadata used for subgroups are missing for many patients (stage for 1,340 of 1,868 test tumours).

## Intended use

Method development and teaching. Any diagnostic use would require prospective validation on independently collected samples.
