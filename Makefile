.PHONY: install run test clean help

# Same commands on every OS; the real work lives in scripts/setup.py.
ifeq ($(OS),Windows_NT)
    VENV_PY = .venv\Scripts\python.exe
    PYTHON ?= python
else
    VENV_PY = .venv/bin/python
    PYTHON ?= python3
endif

install: ## Venv + deps + .env + certs + Piper voices + MT models (add DEV=1 for test deps).
	$(PYTHON) scripts/setup.py $(if $(DEV),--dev,)

run: ## Run the server on https://localhost:8000 (loopback only; BP_HOST=0.0.0.0 for LAN).
	$(VENV_PY) app.py

test: ## Run the backend test suite.
	$(VENV_PY) -m pytest test/hardware_test.py test/vad_tests.py test/mt_model_tests.py test/backend_api_tests.py test/backend_auth_tests.py test/security_tests.py test/config_tests.py -q

clean: ## Remove generated models, certs, venv and caches (never touches speaker_voices/).
	$(PYTHON) -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in ('.venv','.venv-convert','ct2_models','certs','node_modules','.pytest_cache')];[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'
