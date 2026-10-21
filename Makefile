.PHONY: install run test demo-check demo-audio lab clean help

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
	export BP_DISABLE_RATE_LIMIT=1; $(VENV_PY) -m pytest test/hardware_test.py test/vad_tests.py test/mt_model_tests.py test/backend_api_tests.py test/backend_auth_tests.py test/security_tests.py test/config_tests.py -q

clean: ## Remove generated models, certs, venv and caches (never touches speaker_voices/).
	$(PYTHON) -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in ('.venv','.venv-convert','ct2_models','certs','node_modules','.pytest_cache')];[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

demo-check: ## Pre-flight the live demo (assets + the server running on https://localhost:8000).
	python3 scripts/demo_preflight.py --server

demo-audio: ## Process FILE through local pipeline; set SRC/TGT, optional SECONDS=12 VOICE=generic|personal.
	@test -n "$(FILE)" || (echo "Usage: make demo-audio FILE=speaker_voices/clip.m4a SRC=en TGT=sk [SECONDS=12] [VOICE=generic]"; exit 2)
	@test -n "$(SRC)" || (echo "Set SRC=en or SRC=sk"; exit 2)
	@test -n "$(TGT)" || (echo "Set TGT=en or TGT=sk"; exit 2)

lab: ## Voice Lab static page, http://localhost:8080/ui/voice-lab/lab.html, no backend.
	$(VENV_PY) -m http.server 8080 -d ui/voice-lab

clean: ## Remove generated models, certs, venv and caches (never touches speaker_voices/).
	$(PYTHON) -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in ('.venv','.venv-convert','ct2_models','certs','node_modules','.pytest_cache')];[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
