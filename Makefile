PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-bedrock
PLATFORM ?= linux/amd64
DEV_PYTHON ?= 3.11
ARGS=

.PHONY: all install build serve clean lint format test integration_tests docker_build docker_run

all: build

install:
	poetry env use python$(DEV_PYTHON)
	poetry install

build: install
	poetry build

serve: install
	poetry run uvicorn "aidial_adapter_bedrock.app:app" --reload --host "0.0.0.0" --port $(PORT) --workers=1 --env-file ./.env

clean:
	poetry run python -m scripts.clean
	poetry env remove --all

lint: install
	poetry run nox -s lint

format: install
	poetry run nox -s format

test: install
	poetry run nox -s test

integration_tests: install
	poetry run nox -s integration_tests

docker_test:
	docker build --platform $(PLATFORM) -f Dockerfile.test -t $(IMAGE_NAME):test .
	docker run --platform $(PLATFORM) --rm $(IMAGE_NAME):test

docker_serve:
	docker build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .
	docker run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

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