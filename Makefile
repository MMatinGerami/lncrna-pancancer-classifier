.PHONY: all data prepare benchmark budget interpret external figures ablation conformal subgroups curve test lint

all: prepare benchmark budget interpret external figures ablation conformal subgroups curve

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
test:        ; uv run pytest -q
lint:        ; uv run ruff check . && uv run ruff format --check .
