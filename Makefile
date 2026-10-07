.PHONY: all data prepare benchmark budget interpret external figures ablation conformal subgroups curve sites met500 met500-calibration markers site-signal panel microenv test lint docker nextflow nextflow-test

all: prepare benchmark budget interpret external figures ablation conformal subgroups curve sites met500 met500-calibration markers site-signal panel microenv

data:        ; bash scripts/download_data.sh
prepare:     ; uv run python scripts/01_prepare_data.py
benchmark:   ; uv run python scripts/02_benchmark.py
budget:      ; uv run python scripts/03_feature_budget.py
interpret:   ; uv run python scripts/04_interpret.py
external:    ; uv run python scripts/05_external_metastases.py
figures:     ; uv run python scripts/06_figures.py
ablation:    ; uv run python scripts/07_ablation_sex_chromosomes.py
conformal:   ; uv run python scripts/08_conformal.py
subgroups:   ; uv run python scripts/09_subgroups.py
curve:       ; uv run python scripts/10_learning_curve.py
sites:       ; uv run python scripts/11_site_holdout.py
met500:      ; uv run python scripts/12_met500.py
met500-calibration: ; uv run python scripts/13_met500_calibration.py
markers:     ; uv run python scripts/14_marker_stability.py
site-signal: ; uv run python scripts/15_site_signal.py
panel:       ; uv run python scripts/16_sparse_panel.py
microenv:    ; uv run python scripts/17_microenvironment.py
docker:      ; docker build -t lncpan . && docker run --rm -v "$$PWD/data:/app/data" -v "$$PWD/results:/app/results" lncpan make test
nextflow:    ; mamba run -n nextflow nextflow run . -resume
nextflow-test: ; mamba run -n nextflow nextflow run . -profile test
test:        ; uv run pytest -q
lint:        ; uv run ruff check . && uv run ruff format --check .
