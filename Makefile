CONFIG ?= config/default.yaml
PYTHON ?= .venv/bin/python

.PHONY: all extract crosswalk nutrition markets help clean-data

help:
	@echo "nielsen-cereal-data pipeline"
	@echo ""
	@echo "  make venv        Create virtualenv and install dependencies"
	@echo "  make extract     Stage 1: Extract raw TSVs from NielsenIQ tarballs"
	@echo "  make crosswalk   Stage 2a: Build brand crosswalk (2004-2020 <-> 2021+)"
	@echo "  make nutrition    Stage 2b: Parse and assemble nutrition data"
	@echo "  make markets     Stage 3: Aggregate to market-level panel"
	@echo "  make all         Run full pipeline"
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

nutrition: crosswalk
	$(PYTHON) src/nutrition.py --config $(CONFIG)

markets: extract crosswalk
	$(PYTHON) src/aggregate_markets.py --config $(CONFIG) \
		$(if $(YEARS),--years $(YEARS))

all: extract crosswalk nutrition markets
	@echo "Pipeline complete. Outputs in data/clean/"
	@ls -lh data/clean/*.parquet

clean-data:
	rm -rf data/raw/ data/clean/
