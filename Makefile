CONFIG ?= config/default.yaml
PYTHON ?= .venv/bin/python

.PHONY: all extract crosswalk nutrition markets households demographics help clean-data

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
