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

## Unseen sites

Five-fold cross-validation of the training split with every tissue source site confined to one fold (StratifiedGroupKFold) gives macro-F1 0.895 ± 0.013 for the lncRNA model, against 0.938 ± 0.011 with random stratified folds (protein-coding: 0.894 vs 0.937). Performance reported on the random test split therefore overstates what to expect at a new centre by about four points. Within a cancer type, the source site can be predicted from expression with balanced accuracy 0.42 (lncRNA) and 0.37 (protein-coding) against 0.18 by chance (`scripts/15_site_signal.py`).

## Marker stability

Over 30 bootstrap refits of the logistic regression, the ten largest coefficients per cancer type overlap with the full-data list at Jaccard 0.54 on average (0.34 to 0.82); 111 of 330 markers are recovered in at least 90% of resamples and 122 are also SHAP top-10 markers of the XGBoost model (`scripts/14_marker_stability.py`).

## External validation (MET500, Robinson et al. 2017)

437 patients with metastatic biopsies from another institution (poly-A and capture libraries, FPKM quantification). The public matrix contains no lncRNAs from this project's universe, so only the protein-coding model can be evaluated: top-1 accuracy 0.659 (95% CI 0.613 to 0.707), top-3 0.815, similar for poly-A (0.668) and capture (0.629) libraries. Given no informative input, the lncRNA model's conformal sets are all empty (abstention).

On MET500 the protein-coding model is over-confident (ECE 0.126 against 0.012 on the TCGA test split); temperature scaling fitted on TCGA (T = 1.18) reduces it to 0.103. By biopsy site, top-1 accuracy is 0.78 for lymph node (n = 101), 0.78 soft tissue (81), 0.73 bone marrow (44), 0.54 lung (35) and 0.49 liver (123); 37% of liver biopsies are predicted as LIHC or CHOL, the host tissue (`scripts/13_met500_calibration.py`).

## Uncertainty

Split conformal prediction (`scripts/08_conformal.py`, 15% of the training split held out for calibration):

| Setting | Coverage (test) | Coverage (metastases) | Mean set size | Empty sets |
|---|---|---|---|---|
| LAC, α = 0.10, one global threshold | 0.896 | 0.761 | 0.9 | 9.3% |
| LAC, α = 0.10, one threshold per class | 0.918 | 0.945 | 4.9 | 0% |
| LAC, α = 0.05, one global threshold | 0.952 | 0.901 | 1.0 | 2.3% |
| APS, α = 0.10, one global threshold | 0.895 | 1.000 | 14.1 | 10.5% |
| RAPS (k = 1, λ = 0.5), α = 0.10, one global threshold | 0.902 | 0.926 | 0.9 | 6.2% |

An empty set is an abstention; among non-empty LAC sets at α = 0.10, 98.8% contain the true class. RAPS constants are tuned on 30% of the calibration set and the score is calibrated on the remaining 785 tumours. The global threshold loses its guarantee on the metastases (a distribution shift), while per-class thresholds keep it at the cost of larger sets.

## Limitations

- The lncRNA model can only be used where lncRNAs are quantified; MET500, like many clinical pipelines, does not report them.
- Metastases sampled in the liver or lung are often classified as the host organ's cancer; the model cannot separate the primary's signal from the surrounding tissue.
- TCGA only: one consortium, bulk RNA-seq, mostly primary tumours from North American centres. Performance on other platforms, formalin-fixed samples or other populations is unknown.
- Rare classes (fewer than 100 training tumours) are less accurate and less well calibrated.
- Cancers of the same organ (colon vs rectum, oesophagus vs stomach) are not reliably separated; this reflects biology as much as the model.
- Coverage guarantees assume test data are exchangeable with the calibration data, which the metastases result shows does not hold under shift.
- Random test splits share hospital sites with the training data; the held-out-site estimate above is the one to quote for a new centre.
- Clinical metadata used for subgroups are missing for many patients (stage for 1,340 of 1,868 test tumours).

## Intended use

Method development and teaching. Any diagnostic use would require prospective validation on independently collected samples.
