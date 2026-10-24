PYTHON ?= python3

.PHONY: install test smoke all

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/smoke_test.py

all: test smoke
