CONFIG ?= config/default.yaml
PYTHON ?= python3

.PHONY: all extract clean-data help

help:
	@echo "nielsen-cereal-data pipeline"
	@echo ""
	@echo "  make extract     Extract raw TSVs from NielsenIQ tarballs"
	@echo "  make all         Run full pipeline (extract -> clean)"
	@echo ""
	@echo "Override config:  make extract CONFIG=config/custom.yaml"
	@echo "Override years:   make extract YEARS='2014 2018'"

extract:
	$(PYTHON) src/extract.py --config $(CONFIG) $(if $(YEARS),--years $(YEARS))

all: extract
	@echo "TODO: remaining pipeline stages"
