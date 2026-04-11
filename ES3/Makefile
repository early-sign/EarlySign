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

define build_schema
	@echo "Building schema for $(1)..."
	@rm -rf out-$(1) tmp-$(1)
	@npx tsp compile ../earlysign/builtin/$(1)/schema.tsp --emit @typespec/openapi3 --output-dir out-$(1)
	@mkdir -p tmp-$(1)
	@$(DATAMODEL_CODEGEN) \
		--input out-$(1)/@typespec/openapi3/openapi.yaml \
		--output tmp-$(1) \
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
	@$(POETRY) run python ../scripts/fix_distributed_schema.py $(1) tmp-$(1) ../earlysign/builtin/$(1)/schema.py
	@rm -rf tmp-$(1)
endef

$(SENTINEL): $(TSP_SOURCES) Makefile
	@echo "Compiling ES3 Standard..."
	# 1. Compile Core Base Schema
	@echo "Building Core Base Schema..."
	@npx tsp compile schema/es3_v1.tsp --emit @typespec/openapi3 --output-dir out-base
	@rm -rf $(OUTPUT_DIR)
	@$(DATAMODEL_CODEGEN) \
		--input out-base/@typespec/openapi3/openapi.yaml \
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
	
	# 2. Compile each Builtin Schema (Explicit targets for stability)
	$(call build_schema,AVI)
	$(call build_schema,group_sequential)
	$(call build_schema,YEAST)
	
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
