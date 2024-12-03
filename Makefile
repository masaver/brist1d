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
	test -d "$(VENV_PATH)" || python -m venv "$(VENV_PATH)"

## Install
install:
	unzip ./data/data.zip -d ./data/raw
	python -m venv $(VENV_PATH) && source "$(VENV_PATH)/bin/activate" && pip install -r requirements.txt
	$(done)
.PHONY: install

## Update project
update: .activate
	echo "Updating project"
	source "$(VENV_PATH)/bin/activate" && pip install --upgrade -r requirements.txt
	$(done)
.PHONY: update

## Reset project
reset: .activate
	rm -rf $(VENV_PATH)
	$(done)
.PHONY: reset

## List installed packages
list: .activate
	source "$(VENV_PATH)/bin/activate" && pip list
.PHONY: list

## Preprocess data
preprocess: .activate
	source "$(VENV_PATH)/bin/activate" && python src/features/01-extract-patient-data.py
	$(done)
.PHONY: preprocess

## Prepare data
prepare-data: .activate
	source "$(VENV_PATH)/bin/activate" && python src/features/02-prepare-data.py
	$(done)

## Train model 1
train-model-1: .activate
	source "$(VENV_PATH)/bin/activate" && python -m src.models.train_model 1
	$(done)

## Predict model
predict-model-1: .activate
	source "$(VENV_PATH)/bin/activate" && python -m src.models.predict_model 1
	$(done)

## Train model
train-final-model: .activate
	source "$(VENV_PATH)/bin/activate" && python -m src.models.train_model 2
	$(done)

## Predict model
predict-final-model: .activate
	source "$(VENV_PATH)/bin/activate" && python -m src.models.predict_model 2
	$(done)

## Build documentation
build-docs: .activate
	source "$(VENV_PATH)/bin/activate" && jupyter-book clean reports/rendering-1
	source "$(VENV_PATH)/bin/activate" && jupyter-book build reports/rendering-1
	$(done)
.PHONY: doc

## Deploy documentation
deploy-docs:
	source "$(VENV_PATH)/bin/activate" && jupyter-book clean reports/rendering-1
	sed "s/__VERSION__/$(shell git describe --tags --always --dirty=+)/g; s/__DATE__/$(shell git log -1 --format=%cd --date=format:%Y-%m-%d)/g; s/__TIME__/$(shell git log -1 --format=%cd --date=format:%H:%M)/g" reports/rendering-1/_config.yml > reports/rendering-1/_config.yml.tmp
	source "$(VENV_PATH)/bin/activate" && jupyter-book build reports/rendering-1 --config reports/rendering-1/_config.yml.tmp
	rm reports/rendering-1/_config.yml.tmp
	rsync -avz --delete --exclude={'.htaccess','.htpasswd'} reports/rendering-1/_build/html/ ssh-w0190139@w0190139.kasserver.com:/www/htdocs/w0190139/brist1d.junghanns.it/
	$(done)
.PHONY: deploy-docs
