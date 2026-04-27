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
	PYTHONPATH=src $(PYTHON) scripts/run_all_candidates.py --lost-embed data/processed/embeddings_synthetic_v2.npy --output-dir reports/tables

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
paper:
	pdflatex -interaction=nonstopmode submersion_espectral_lenguajes_perdidos_paper.tex
	pdflatex -interaction=nonstopmode submersion_espectral_lenguajes_perdidos_paper.tex

notebook:
	jupyter notebook notebooks/

lint:
	black src/ scripts/ tests/
