.PHONY: compile install clean check-sync

POETRY ?= poetry
# Support 3.11 to cover minor typing discrepancies between 3.12 and 3.11
TARGET_PYTHON_VERSION ?= 3.11
DATAMODEL_CODEGEN ?= $(POETRY) run datamodel-codegen
OUTPUT_DIR ?= ../earlysign/schema/ES3
SENTINEL ?= $(OUTPUT_DIR)/__init__.py
TSP_SOURCE ?= build.tsp

TSP_SOURCES := schema/es3_v1.tsp $(shell find ../earlysign/builtin -name '*.tsp')

install:
	npm install

compile: $(SENTINEL)

$(SENTINEL): $(TSP_SOURCES) Makefile
	@echo "Compiling ES3 Standard..."
	# 0. Generate build.tsp dynamically
	@echo 'import "./schema/es3_v1.tsp";' > $(TSP_SOURCE)
	@for f in $$(find ../earlysign/builtin -name '*.tsp'); do \
		echo "import \"$$f\";" >> $(TSP_SOURCE); \
	done
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
		--target-python-version $(TARGET_PYTHON_VERSION) \
		--keep-model-order \
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
	rm -rf out-openapi3 node_modules tsp-output package-lock.json
