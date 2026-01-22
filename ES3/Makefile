.PHONY: compile install clean check-sync

POETRY ?= poetry
TARGET_PYTHON_VERSION ?= 3.12
DATAMODEL_CODEGEN ?= $(POETRY) run datamodel-codegen
OUTPUT_DIR ?= ../earlysign/schema/ES3
SENTINEL ?= $(OUTPUT_DIR)/__init__.py
TSP_SOURCE ?= schema/es3_v1.tsp

install:
	npm install

compile: $(SENTINEL)

$(SENTINEL): $(TSP_SOURCE)
	@echo "Compiling ES3 Standard..."
	# 1. Compile TypeSpec to OpenAPI 3
	npx tsp compile $(TSP_SOURCE) --emit @typespec/openapi3 --output-dir out-openapi3
	# 2. Generate Pydantic models
	rm -rf $(OUTPUT_DIR)
	$(DATAMODEL_CODEGEN) \
		--input out-openapi3/@typespec/openapi3/openapi.yaml \
		--output $(OUTPUT_DIR) \
		--input-file-type openapi \
		--output-model-type pydantic_v2.BaseModel \
		--base-class pydantic.BaseModel \
		--use-schema-description \
		--treat-dot-as-module \
		--target-python-version $(TARGET_PYTHON_VERSION) \
		--reuse-model \
		--use-type-alias \
		--use-double-quotes \
		--openapi-scopes schemas \
		--enum-field-as-literal one \
		--capitalise-enum-members \
		--use-one-literal-as-default \
		--use-standard-collections \
		--disable-timestamp
	@touch $(SENTINEL)

check-sync:
	@echo "Checking if ES3 SDK is in sync with TypeSpec source..."
	@if [ $(TSP_SOURCE) -nt $(SENTINEL) ]; then \
		echo "Error: ES3 SDK is out of sync with $(TSP_SOURCE). Please run 'make compile-ES3'."; \
		exit 1; \
	else \
		echo "ES3 SDK is in sync."; \
	fi

clean:
	rm -rf out-openapi3 node_modules tsp-output package-lock.json
