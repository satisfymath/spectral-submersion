# Spectral Submersion: full pipeline Makefile
# Usage:
#   make setup          # install dependencies
#   make pipeline       # run the complete benchmark pipeline
#   make test           # run tests
#   make paper          # compile LaTeX paper

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := src

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install tabulate

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v

# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------
build-dataset:
	PYTHONPATH=src $(PYTHON) scripts/build_dataset.py

candidates:
	# Download and build candidate corpora from OPUS-Tatoeba
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/maori_tatoeba_raw.txt --output data/raw/candidate_languages/maori_tokens.csv
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/tahitian_tatoeba_raw.txt --output data/raw/candidate_languages/tahitian_tokens.csv
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/haw_tatoeba_raw.txt --output data/raw/candidate_languages/haw_tokens.csv
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/sm_tatoeba_raw.txt --output data/raw/candidate_languages/sm_tokens.csv
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/to_tatoeba_raw.txt --output data/raw/candidate_languages/to_tokens.csv
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/fj_tatoeba_raw.txt --output data/raw/candidate_languages/fj_tokens.csv
	PYTHONPATH=src $(PYTHON) scripts/build_candidate_corpus.py --input data/external/rap_tatoeba_raw.txt --output data/raw/candidate_languages/rap_tokens.csv

segment:
	PYTHONPATH=src $(PYTHON) scripts/segment_polynesian.py

# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
embed-lost:
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/processed/lost_tokens.csv --output data/processed/embeddings_lost.npy --sv-output data/processed/singular_values_lost.npy --fig reports/figures/singular_values_lost.png

embed-synthetic:
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/lost_language/corpus_synthetic_v2.csv --output data/processed/embeddings_synthetic_v2.npy --sv-output data/processed/singular_values_synthetic_v2.npy --fig reports/figures/singular_values_synthetic_v2.png --k 32

embed-candidates:
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/maori_tokens.csv --output data/processed/embeddings_mi.npy --sv-output data/processed/singular_values_mi.npy --fig reports/figures/singular_values_mi.png
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/tahitian_tokens.csv --output data/processed/embeddings_ty.npy --sv-output data/processed/singular_values_ty.npy --fig reports/figures/singular_values_ty.png
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/haw_tokens.csv --output data/processed/embeddings_haw.npy --sv-output data/processed/singular_values_haw.npy --fig reports/figures/singular_values_haw.png
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/sm_tokens.csv --output data/processed/embeddings_sm.npy --sv-output data/processed/singular_values_sm.npy --fig reports/figures/singular_values_sm.png
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/to_tokens.csv --output data/processed/embeddings_to.npy --sv-output data/processed/singular_values_to.npy --fig reports/figures/singular_values_to.png
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/fj_tokens.csv --output data/processed/embeddings_fj.npy --sv-output data/processed/singular_values_fj.npy --fig reports/figures/singular_values_fj.png
	PYTHONPATH=src $(PYTHON) scripts/build_embeddings.py --input data/raw/candidate_languages/rap_tokens.csv --output data/processed/embeddings_rap.npy --sv-output data/processed/singular_values_rap.npy --fig reports/figures/singular_values_rap.png

# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
freq:
	PYTHONPATH=src $(PYTHON) scripts/run_frequency_analysis.py

svd:
	PYTHONPATH=src $(PYTHON) scripts/run_svd.py

negative-controls:
	PYTHONPATH=src $(PYTHON) scripts/run_negative_controls.py --input data/raw/lost_language/corpus_synthetic_v2.csv --output reports/tables/control_comparison_v2.csv

bootstrap:
	PYTHONPATH=src $(PYTHON) scripts/run_bootstrap.py --input data/raw/lost_language/corpus_synthetic_v2.csv --output reports/tables/bootstrap_stability.csv --n-bootstrap 50

# ------------------------------------------------------------------
# Alignment and validation
# ------------------------------------------------------------------
align-synthetic:
	PYTHONPATH=src $(PYTHON) scripts/run_alignment.py --candidate-embed data/processed/embeddings_synthetic_candidate.npy --candidate-vocab data/processed/embeddings_synthetic_candidate.vocab.json --candidate-name synthetic --reg 0.5

validate-anchors:
	PYTHONPATH=src $(PYTHON) scripts/validate_anchors.py --train-fraction 0.20 --seed 42

validate-polysemy:
	PYTHONPATH=src $(PYTHON) scripts/validate_polysemy.py --train-fraction 0.20 --seed 42

compare-all:
	PYTHONPATH=src $(PYTHON) scripts/run_diverse_candidates.py --lost-embed data/processed/embeddings_synthetic_v2.npy --output reports/tables/diverse_comparison_synthetic.csv

compare-indus:
	PYTHONPATH=src $(PYTHON) scripts/run_diverse_candidates.py --lost-embed data/processed/embeddings_indus_real.npy --output reports/tables/diverse_comparison_indus_real.csv

# ------------------------------------------------------------------
# Additional analyses
# ------------------------------------------------------------------
entropy-analysis:
	PYTHONPATH=src $(PYTHON) scripts/analyze_conditional_entropy.py --input data/raw/lost_language/corpus_indus_real.csv --output reports/tables/entropy_analysis_indus_real.csv
	PYTHONPATH=src $(PYTHON) scripts/analyze_conditional_entropy.py --input data/raw/lost_language/corpus_synthetic.csv --output reports/tables/entropy_analysis_synthetic.csv

corpus-comparison:
	PYTHONPATH=src $(PYTHON) scripts/visualize_corpus_comparison.py --output reports/figures/corpus_comparison.png

master-summary:
	PYTHONPATH=src $(PYTHON) scripts/generate_master_summary.py --output reports/final/master_experiment_summary.md

# ------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------
report:
	PYTHONPATH=src $(PYTHON) scripts/generate_report.py

integrated-report:
	PYTHONPATH=src $(PYTHON) scripts/generate_integrated_report.py

# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------
pipeline: test embed-synthetic negative-controls bootstrap validate-anchors validate-polysemy compare-all integrated-report
	@echo "Full pipeline complete. See reports/final/integrated_hypothesis_report.md"

# ------------------------------------------------------------------
# Paper
# ------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PhD upgrade: auditable pipeline targets
# ---------------------------------------------------------------------------
phd-stability:
	PYTHONPATH=src $(PYTHON) scripts/run_phd_stability.py --config configs/phd_upgrade.yaml

phd-experiments:
	PYTHONPATH=src $(PYTHON) scripts/run_phd_experiments.py --config configs/phd_upgrade.yaml

phd-ledger:
	PYTHONPATH=src $(PYTHON) scripts/run_phd_ledger.py --config configs/phd_upgrade.yaml

phd-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_phd_audit.py --config configs/phd_upgrade.yaml

phd-audit-v2:
	PYTHONPATH=src $(PYTHON) scripts/run_phd_audit_v2.py --config configs/phd_upgrade.yaml

# ---------------------------------------------------------------------------
# Reproducible run (Section 38-40)
# ---------------------------------------------------------------------------
RUN_ID := $(shell date +%Y%m%d_%H%M%S)
RUN_DIR := runs/$(RUN_ID)

run-init:
	mkdir -p $(RUN_DIR)
	git rev-parse HEAD > $(RUN_DIR)/git_commit.txt 2>/dev/null || echo "unknown" > $(RUN_DIR)/git_commit.txt
	cp configs/phd_upgrade.yaml $(RUN_DIR)/config.yaml
	PYTHONPATH=src $(PYTHON) -m pip freeze > $(RUN_DIR)/environment.yml 2>/dev/null || true
	echo '{"seed": 42, "run_id": "$(RUN_ID)"}' > $(RUN_DIR)/random_seeds.json

reproduce_all: run-init test phd-stability phd-experiments phd-ledger phd-audit
	@echo "Reproducible pipeline complete. See $(RUN_DIR)/"

# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------
paper:
	pdflatex -interaction=nonstopmode submersion_espectral_lenguajes_perdidos_paper.tex
	pdflatex -interaction=nonstopmode submersion_espectral_lenguajes_perdidos_paper.tex

notebook:
	jupyter notebook notebooks/

lint:
	black src/ scripts/ tests/
