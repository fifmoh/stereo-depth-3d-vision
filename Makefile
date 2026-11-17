PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
SCENES := Adirondack Backpack Bicycle1
SCENES_RECT := Adirondack Backpack

.PHONY: help venv install data data-full test demo demo-live pdf lint clean

help:
	@echo "make install   create .venv and install requirements"
	@echo "make data      download a starter subset of Middlebury 2014 (no login)"
	@echo "make data-full download every split (~700 MB)"
	@echo "make test      run the pytest suite (offline smoke test included)"
	@echo "make demo      offline demo on skimage.data.stereo_motorcycle()"
	@echo "make pdf       build docs/RAPPORT.pdf from docs/RAPPORT.md (pandoc + tectonic)"
	@echo "make clean     remove caches and generated outputs"

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: venv
	$(BIN)/pip install -r requirements.txt

data:
	$(BIN)/python scripts/fetch_data.py $(addprefix --scene ,$(SCENES)) \
	  $(addprefix --imperfect ,$(SCENES_RECT)) --tutorial

data-full:
	$(BIN)/python scripts/fetch_data.py --split tune --split val --split test \
	  --split rect --tutorial

test:
	PYTHONPATH=src $(BIN)/python -m pytest tests -q

demo:
	PYTHONPATH=src $(BIN)/python -m stereo_depth.cli disparity \
	  --scene data/raw/Adirondack-perfect --scale 0.5

demo-live:
	PYTHONPATH=src $(BIN)/python demo/live_demo.py

pdf:
	pandoc docs/RAPPORT.md -o docs/RAPPORT.pdf \
	  --pdf-engine=tectonic --toc --number-sections \
	  --resource-path=docs -V lang=fr -V geometry:margin=2.5cm

lint:
	$(BIN)/python -m compileall -q src scripts tests

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf outputs/figures/* outputs/metrics/* outputs/pointclouds/*
