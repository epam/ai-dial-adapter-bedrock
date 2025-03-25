PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-bedrock
PLATFORM ?= linux/amd64
DEV_PYTHON ?= 3.11
DOCKER ?= docker
VENV_DIR ?= .venv
POETRY ?= $(VENV_DIR)/bin/poetry
POETRY_VERSION ?= 1.8.5
ARGS ?=

.PHONY: all init_env install build serve clean lint format test integration_tests docker_build docker_run

all: build

init_env:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install poetry==$(POETRY_VERSION) --quiet

install: init_env
	$(POETRY) env use python$(DEV_PYTHON)
	$(POETRY) install

build: install
	$(POETRY) build

serve: install
	$(POETRY) run uvicorn "aidial_adapter_bedrock.app:app" --reload --host "0.0.0.0" --port $(PORT) --workers=1 --env-file ./.env

clean:
	$(POETRY) run python -m scripts.clean
	$(POETRY) env remove --all

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run -- nox -s test -- $(ARGS)

integration_tests: install
	$(POETRY) run -- nox -s integration_tests -- $(ARGS)

docker_test:
	$(DOCKER) build --platform $(PLATFORM) -f Dockerfile.test -t $(IMAGE_NAME):test .
	$(DOCKER) run --platform $(PLATFORM) --rm $(IMAGE_NAME):test

docker_serve:
	$(DOCKER) build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .
	$(DOCKER) run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

help:
	@echo '===================='
	@echo 'build                        - build the source and wheels archives'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'docs                         - build the documentation'
	@echo 'publish'                     - publish the Docker image to the registry'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
	@echo 'docker_serve                 - run the dev server from the docker'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test ARGS=<test_file>        - run all tests in file'
	@echo 'docker_test                  - run unit tests from the docker'
	@echo 'integration_tests            - run integration tests'