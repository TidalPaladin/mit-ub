.PHONY: build benchmark clean clean-env check quality style tag-version test env upload upload-test

PROJECT=mit_ub
QUALITY_DIRS=$(PROJECT) tests scripts
CLEAN_DIRS=$(PROJECT) tests scripts
PYTHON=pdm run python

IMAGE_NAME=tidalpaladin/mit-ub
TAG=latest

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test

clean: ## remove cache files
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete

clean-env: ## remove the virtual environment directory
	pdm venv remove $(PROJECT)


deploy: ## installs from lockfile
	git submodule update --init --recursive
	which pdm || pip install --user pdm
	pdm venv create -n $(PROJECT)-deploy
	pdm install --production --no-lock


init: ## pulls submodules and initializes virtual environment
	git submodule update --init --recursive
	which pdm || pip install --user pdm
	pdm venv create
	pdm install -d -G train --no-isolation
	$(PYTHON) -m pip install "transformer-engine[pytorch]" --no-build-isolation

node_modules: 
ifeq (, $(shell which npm))
	$(error "No npm in $(PATH), please install it to run pyright type checking")
else
	npm install
endif

quality:
	$(MAKE) clean
	$(PYTHON) -m black --check $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a $(QUALITY_DIRS)

style:
	$(PYTHON) -m autoflake -r -i $(QUALITY_DIRS)
	$(PYTHON) -m isort $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a $(QUALITY_DIRS)
	$(PYTHON) -m black $(QUALITY_DIRS)
	find csrc/ -type f \( -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +

test: ## run unit tests
	$(PYTHON) -m pytest \
		-rs \
		--cov=./$(PROJECT) \
		--cov-report=term \
		./tests/

test-%: ## run unit tests matching a pattern
	$(PYTHON) -m pytest -s -r fE -k $* ./tests/ --tb=no

test-pdb-%: ## run unit tests matching a pattern with PDB fallback
	$(PYTHON) -m pytest -rs --pdb -k $* -v ./tests/ 

test-ci: ## runs CI-only tests
	$(PYTHON) -m pytest \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		-s -v \
		-m "not ci_skip" \
		./tests/

types: node_modules
	pdm run npx --no-install pyright tests $(PROJECT)

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
