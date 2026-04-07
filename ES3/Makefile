.PHONY: compile install clean check-sync

POETRY ?= poetry
# Support 3.11 to cover minor typing discrepancies between 3.12 and 3.11
TARGET_PYTHON_VERSION ?= 3.11
DATAMODEL_CODEGEN ?= $(POETRY) run datamodel-codegen
OUTPUT_DIR ?= ../earlysign/schema/ES3
SENTINEL ?= $(OUTPUT_DIR)/__init__.py
TSP_SOURCE ?= build.tsp

# Unified sources
TSP_SOURCES := schema/es3_v1.tsp $(shell find ../earlysign/builtin -name 'schema.tsp')

install:
	npm install

compile: $(SENTINEL)

$(SENTINEL): $(TSP_SOURCES) Makefile
	@echo "Compiling ES3 Standard..."
	# 1. Compile Core Base Schema
	@echo "Building Core Base Schema..."
	@mkdir -p out-base
	@npx tsp compile schema/es3_v1.tsp --emit @typespec/openapi3 --output-dir out-base
	@cp out-base/@typespec/openapi3/openapi.yaml ../base_openapi.yaml
	@rm -rf $(OUTPUT_DIR)
	@$(DATAMODEL_CODEGEN) \
		--input ../base_openapi.yaml \
		--output $(OUTPUT_DIR) \
		--input-file-type openapi \
		--output-model-type pydantic_v2.BaseModel \
		--snake-case-field \
		--use-schema-description \
		--use-field-description \
		--target-python-version $(TARGET_PYTHON_VERSION) \
		--disable-timestamp \
		--use-double-quotes \
		--output-datetime-class AwareDatetime \
		--capitalise-enum-members \
		--enum-field-as-literal one \
		--use-one-literal-as-default \
		--use-type-alias
	
	# Fix absolute imports for Core SDK
	@$(POETRY) run python ../scripts/fix_imports_absolute.py $(OUTPUT_DIR) earlysign.schema.ES3 false
	
	# 2. Compile each Builtin Schema using the modular orchestrator
	@echo "Building Builtin Modular SDKs..."
	@$(POETRY) run python ../scripts/generate_modular_sdk.py AVI group_sequential YEAST
	
	@touch $(SENTINEL)

check-sync:
	@echo "Checking if ES3 SDK is in sync with TypeSpec source..."
	@needs_update=0; \
	for file in $(TSP_SOURCES); do \
		if [ "$$file" -nt $(SENTINEL) ]; then \
			echo "Error: ES3 SDK is out of sync with $$file. Please run 'make compile-ES3'."; \
			needs_update=1; \
			break; \
		fi; \
	done; \
	if [ $$needs_update -eq 1 ]; then \
		exit 1; \
	else \
		echo "ES3 SDK is in sync."; \
	fi

clean:
	rm -rf out-* build.tsp
