ENV ?= DEV
PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-bedrock

ifeq ($(ENV),DEV)
	REQ = requirements-dev.txt
else ifeq ($(ENV),PROD)
	REQ = requirements.txt
endif

VENV=.venv
BUILD=$(VENV)/bin/activate

install: $(BUILD)

$(BUILD): requirements.txt $(REQ)
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r $(REQ)
	@echo "\033[31mActivate venv by running:\n> source $(BUILD)\033[0m"

.PHONY: install server-run client-run clean lint format test integration_test docker-build docker-run

server-run: $(BUILD)
	python -m server.main --port=$(PORT)

client-run: $(BUILD)
	@source ./load_env.sh; load_env; \
	python -m client.client_adapter

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete

lint: $(BUILD)
	pyright
	flake8
	$(MAKE) format ARGS="--check"

format: $(BUILD)
	autoflake . $(ARGS)
	isort . $(ARGS)
	black . $(ARGS)

TEST_FILES ?= ./tests/unit_tests

# Add options "-s --log-cli-level=NOTSET" to pytest to see all logs
test: $(BUILD)
	python -m pytest $(TEST_FILES) -v --durations=0 -rA

integration_test: $(BUILD)
	@source ./load_env.sh; load_env; \
	python -m pytest ./tests/integration_tests -v --durations=0 -rA $(ARGS)

docker-build: Dockerfile
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .

docker-run: docker-build
	docker run --platform linux/amd64 --env-file ./.env -p $(PORT):5000 $(IMAGE_NAME)