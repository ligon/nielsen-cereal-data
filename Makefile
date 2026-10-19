CONFIG ?= config/default.yaml
PYTHON ?= .venv/bin/python

.PHONY: all extract crosswalk nutrition markets households demographics help clean-data rms rms-test rms-validate

help:
	@echo "nielsen-cereal-data pipeline"
	@echo ""
	@echo "  make venv        Create virtualenv and install dependencies"
	@echo "  make extract      Stage 1: Extract raw TSVs from NielsenIQ tarballs"
	@echo "  make crosswalk    Stage 2: Build brand crosswalk (2004-2020 <-> 2021+)"
	@echo "  make nutrition    Stage 3: Parse and assemble nutrition data"
	@echo "  make markets      Stage 4: Aggregate to market-level panel"
	@echo "  make households   Stage 5: Aggregate to household-level panel"
	@echo "  make demographics Stage 6: Compute demographic aggregates"
	@echo "  make all          Run full pipeline"
	@echo "  make rms          RMS second source: per-year parquet (D2) + market_panel_rms (D3)"
	@echo "  make rms-test     D3 contract test for market_panel_rms.parquet"
	@echo "  make rms-validate D4 validation report, RMS vs panel -> docs/rms-validation.org"
	@echo ""
	@echo "Override config:   make all CONFIG=config/custom.yaml"
	@echo "Override years:    make extract YEARS='2014 2018'"
	@echo "Parallel extract:  make extract JOBS=8"

venv:
	python3.11 -m venv .venv
	.venv/bin/pip install -q pandas pyarrow pyyaml requests

extract:
	$(PYTHON) src/extract.py --config $(CONFIG) \
		$(if $(YEARS),--years $(YEARS)) \
		$(if $(JOBS),-j $(JOBS))

crosswalk: extract
	$(PYTHON) src/brand_crosswalk.py --config $(CONFIG)

nutrition: extract
	$(PYTHON) src/nutrition.py --config $(CONFIG)

markets: extract crosswalk
	$(PYTHON) src/aggregate_markets.py --config $(CONFIG) \
		$(if $(YEARS),--years $(YEARS))

households: extract crosswalk
	$(PYTHON) src/aggregate_households.py --config $(CONFIG) \
		$(if $(YEARS),--years $(YEARS))

demographics: extract
	$(PYTHON) src/demographics.py --config $(CONFIG) \
		$(if $(YEARS),--years $(YEARS))

# crosswalk and nutrition are independent; `make -j` runs them in parallel.
# households and demographics are independent of each other.
all: extract crosswalk nutrition markets households demographics
	@echo "Pipeline complete. Outputs in data/clean/"
	@ls -lh data/clean/*.parquet

clean-data:
	rm -rf data/raw/ data/clean/

# --- RMS (retail scanner) second source; see src/rms_build.sh -------------
# Independent of the Homescan targets above; writes only under
# Data/NielsenScanner/rms (group-only) and docs/rms-d{2,3}-log.org.
# RMS_PYTHON defaults to the CerealDemand venv (pandas 3.0.3) so the D3 parquet
# and its test share one pandas.
RMS_PYTHON ?= /global/scratch/fsa/fc_jevons/ligon/mirrors/CerealDemand/.venv/bin/python

rms:
	PYTHON=$(RMS_PYTHON) src/rms_build.sh all

rms-test:
	$(RMS_PYTHON) -m pytest -q -p no:cacheprovider tests/test_rms_market_panel.py

# D4: reads market_panel_rms.parquet, rms/diagnostics/*, the panel, and (if
# present) rms/laws/homog_compare.json from CerealDemand src/rms_homog_compare.py.
rms-validate:
	$(RMS_PYTHON) src/rms_validate.py
