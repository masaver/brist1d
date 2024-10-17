.SILENT:
.PHONY: help

SHELL=/bin/bash

done = printf "\e[32m ✔ Done\e[0m\n\n";

## This help screen
help:
	printf "Available commands\n\n"
	awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "\033[33m%-40s\033[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

PROJECT = brist1d
VENV_PATH = .venv

.activate:
	test -d "$(VENV_PATH)" || python -m venv "$(VENV_PATH)" && source "$(VENV_PATH)/bin/activate"

## Install
install:
	unzip ./data/data.zip -d ./data/raw
	python -m venv $(VENV_PATH) && source "$(VENV_PATH)/bin/activate" && pip install -r requirements.txt
	$(done)
.PHONY: install

## Update project
update: .activate
	echo "Updating project"
	pip install --upgrade -r requirements.txt
	$(done)
.PHONY: update

## Reset project
reset: .activate
	rm -rf $(VENV_PATH)
	$(done)
.PHONY: reset

## List installed packages
list: .activate
	pip list
.PHONY: list

## Preprocess data
preprocess: .activate
	python src/features/01-extract-patient-data.py
	$(done)
.PHONY: preprocess

## Build documentation
build-docs: .activate
	jupyter-book clean  reports/rendering-1
	jupyter-book build reports/rendering-1
	$(done)
.PHONY: doc
