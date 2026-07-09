PYTHON = .venv/bin/python3
SHELL := /bin/bash


#check if python exist in venv, otherwise fallback to default
ifeq ("$(PYTHON)",".venv/bin/python3")
ifeq ("$(wildcard ${PYTHON})","")
$(warning Python not found in .venv, falling back to default)
PYTHON=python3
endif
else 
$(info Using python: $(PYTHON))
endif

REQUIRED_PYTHON_VERSION := 3.8
PYTHON_VERSION := $(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

version_check = $(shell python3 -c 'import sys; required_version = tuple(map(int, "$(REQUIRED_PYTHON_VERSION)".split("."))); \
    sys.exit((sys.version_info.major, sys.version_info.minor) < required_version)')

# Terminate the make process if the version is below the required version
ifeq ($(version_check),1)
$(error Python $(REQUIRED_PYTHON_VERSION) or higher is required. Installed version: $(PYTHON_VERSION))
else
$(info Python version is satisfied: $(PYTHON_VERSION))
endif


all: install  

# Installs without external dependencies
install: clean venv ftio_venv msg 

# Installs with external dependencies
full: venv ftio_venv_full msg

# Installs debug version external dependencies
debug: venv ftio_debug_venv msg 

ftio_venv: override PYTHON = .venv/bin/python3
ftio_venv: ftio

ftio_venv_full: override PYTHON = .venv/bin/python3
ftio_venv_full: ftio_full

ftio_debug_venv: override PYTHON = .venv/bin/python3
ftio_debug_venv: ftio_debug

ftio_debug: 
	$(PYTHON) -m pip install -e '.[external-libs,development-libs,plot-libs]' --no-cache-dir || \
	(echo "Installing external libs failed, trying fallback..." && $(PYTHON) -m pip install -e . --no-cache-dir)

ftio: 
	$(PYTHON) -m pip install . 

ftio_full: 
	$(PYTHON) -m pip install -e '.[external-libs,development-libs,plot-libs,amd-libs,ml-libs]' --no-cache-dir || \
	(echo "Installing external libs failed, trying fallback..." && $(PYTHON) -m pip install -e . --no-cache-dir)
venv: 
	$(PYTHON) -m venv .venv 
	@echo -e "Environment created. Using python from .venv/bin/python3" 

msg: 
	@echo -e "\nftio was installed in an python environment in .venv" 
	@echo -e "To activate python from this venv, call:\nsource $(PWD)/.venv/bin/activate\n"
	@echo -e "Afterwards, you can just call 'ftio [filename]'"
#########



clean_project:
	echo "Cleaning old installation"
	$(PYTHON) -m pip uninstall --yes ftio-hpc || echo "no installation of ftio found"
	# @mv old_pyproject pyproject.toml && mv setup.py old_setup || true

clean: clean_project
	rm -rf .venv || true


docker:
	docker build -f docker/Dockerfile -t freq_io:1.0 .


docker_run:
	docker run -v "$$PWD/examples/tmio/JSONL/8.jsonl:/freq_io/8.jsonl" -t freq_io:1.0 8.jsonl -e no


docker_interactive:
	docker run -ti --entrypoint /bin/bash freq_io:1.0



# profile 
profile:
	rm -f test.pstats
	$(PYTHON) -m cProfile -o test.pstats ftio/cli/ftio_core.py -h
	$(PYTHON) -m pip install snakeviz
	snakeviz test.pstats

profile2:
	$(PYTHON) -m pip install pyinstrument
	$(PYTHON) -m pyinstrument ftio/cli/ftio_core.py  -h

# test
test_all:
	mkdir quicktest
	cp examples/tmio/JSONL/8.jsonl quicktest
	@cd quicktest && ftio 8.jsonl -e no && echo "--- passed ftio ---" || echo "--- failed ftio ---"
	@cd quicktest && ftio 8.jsonl -e no -o dbscan && echo "--- passed ftio ---" || echo "--- failed ftio ---"
	@cd quicktest && ftio 8.jsonl -e no -o lof && echo "--- passed ftio ---" || echo "--- failed ftio ---"
	@cd quicktest && ioparse 8.jsonl && echo "--- passed ioparse ---" || echo "--- failed ioparse ---"
	@cd quicktest && ioplot 8.jsonl --no_disp && echo "--- passed ioplot ---" || echo "--- failed ioplot ---"
	@rm -rf ./quicktest/*

test:
	cd test && python3 -m pytest -v && make clean

test_parallel:
	@python3 -m pip show pytest-xdist > /dev/null 2>&1 || python3 -m pip install pytest-xdist
	cd test && python3 -m pytest -n 4 && make clean

test_failed:
	cd test && python3 -m pytest --ff && make clean

check_style: check_tools
	black .
	ruff check --fix
# 	flake8 .

check_tools:
	@command -v black >/dev/null 2>&1 || (echo "black not found, installing..." && pip install black)
	@command -v isort >/dev/null 2>&1 || (echo "isort not found, installing..." && pip install isort)
	@command -v flake8 >/dev/null 2>&1 || (echo "flake8 not found, installing..." && pip install flake8)


quick_test:
	$(PWD)/.venv/bin/ftio -e no -h 


# publish
testpypi: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

testpypi-install:	
	$(PYTHON) -m pip install --index-url https://test.pypi.org/simple/ --no-deps ftio_hpc

pypi: build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*
	$(PYTHON) -m pip install ftio_hpc

build: pack

pack:
	$(PYTHON) -m  pip install --upgrade pip
	$(PYTHON) -m  pip install --upgrade build 
	$(PYTHON) -m build



.PHONY: all test test_all clean clean_all build pack testpypi ftio
