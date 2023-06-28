BASENAME=server
CONTAINER_NAME=ghcr.io/annotation-ai/$(BASENAME):latest
PYTHON=3.9
FORMAT_OPTION=-m 100 -t 4 -w
FORMAT_TARGET=*.go tritonserver/*.go

# For users
setup:
	echo "Nothing to setup"

.PHONY: build
build:
	go build main.go

run:
	$(MAKE) build
	./main

run-profile:
	$(MAKE) build
	./main --profile

# docker
docker-build:
	docker build -t $(CONTAINER_NAME) .

docker-pull:
	docker pull $(CONTAINER_NAME)

docker-push:
	docker push $(CONTAINER_NAME)

docker-run:
	docker run -it --rm $(CONTAINER_NAME)

# For developers
env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup-dev:
	sh setup-dev.sh

setup-python:
	pip install -r requirements-dev.txt
	pre-commit install

.PHONY: format
format:
	golines $(FORMAT_OPTION) $(FORMAT_TARGET)
	swag fmt

format-check:
	$(eval ListToFix :=$(shell golines $(FORMAT_OPTION) -l $(FORMAT_TARGET)))
	$(if $(strip $(ListToFix)),exit 1,echo passed)

.PHONY: docs
docs:
	swag init

lint:
	golangci-lint run


