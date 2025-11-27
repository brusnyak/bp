.PHONY: install install-deps run test certs clean help

# Detect operating system
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

# Python version
PYTHON_VERSION = 3.11

# Platform-specific variables
ifeq ($(UNAME_S),Darwin)  # macOS
    VENV_NAME = venv
    PYTHON_EXEC = $(VENV_NAME)/bin/python$(PYTHON_VERSION)
    PIP_EXEC = $(VENV_NAME)/bin/pip
    DYLD_ENV = DYLD_LIBRARY_PATH="/opt/homebrew/lib"
    FFMPEG_INSTALL = brew install ffmpeg || true
else ifeq ($(UNAME_S),Linux)
    VENV_NAME = venv
    PYTHON_EXEC = $(VENV_NAME)/bin/python$(PYTHON_VERSION)
    PIP_EXEC = $(VENV_NAME)/bin/pip
    DYLD_ENV = 
    FFMPEG_INSTALL = @echo "Please install FFmpeg: sudo apt-get install ffmpeg (Debian/Ubuntu) or sudo yum install ffmpeg (RHEL/CentOS)"
else  # Windows (assumes running in Git Bash or WSL)
    VENV_NAME = venv
    PYTHON_EXEC = $(VENV_NAME)/Scripts/python.exe
    PIP_EXEC = $(VENV_NAME)/Scripts/pip.exe
    DYLD_ENV = 
    FFMPEG_INSTALL = @echo "Please install FFmpeg manually or use Chocolatey: choco install ffmpeg"
endif

install: ## Install all Python and Node.js dependencies, including models.
	@echo "--- Creating Python virtual environment '$(VENV_NAME)' with Python $(PYTHON_VERSION) ---"
	python$(PYTHON_VERSION) -m venv $(VENV_NAME)
	@echo "--- Installing all Python project dependencies from requirements.txt ---"
	$(PIP_EXEC) install -r requirements.txt
	@echo "--- Installing FFmpeg (platform-specific) ---"
	$(FFMPEG_INSTALL)
	@echo "--- Installing Coqui TTS ---"
	$(PIP_EXEC) install TTS
	@echo "--- Downloading MT models (CTranslate2) ---"
	$(PYTHON_EXEC) -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-en-sk', 'ct2_models/Helsinki-NLP--opus-mt-en-sk', quantization='int8')"
	$(PYTHON_EXEC) -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-sk-en', 'ct2_models/Helsinki-NLP--opus-mt-sk-en', quantization='int8')"
	$(PYTHON_EXEC) -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as converter; converter.convert_model('Helsinki-NLP/opus-mt-en-cs', 'ct2_models/Helsinki-NLP--opus-mt-en-cs', quantization='int8')"
	@echo "--- Downloading Piper TTS models ---"
	$(PYTHON_EXEC) backend/tts/download_piper_models.py en_US-ryan-medium
	$(PYTHON_EXEC) backend/tts/download_piper_models.py sk_SK-lili-medium
	$(PYTHON_EXEC) backend/tts/download_piper_models.py cs_CZ-jirka-medium
	@echo "--- Installing frontend dependencies ---"
	cd frontend && npm install
	@echo "--- Installation complete. ---"

run: ## Run the FastAPI backend server.
	$(DYLD_ENV) TORCH_USE_LIBAV=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONHASHSEED=random $(PYTHON_EXEC) app.py

test: ## Run the comprehensive backend test suite.
	$(DYLD_ENV) TORCH_USE_LIBAV=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONHASHSEED=random $(PYTHON_EXEC) test/piper_pipeline_test.py
	$(DYLD_ENV) TORCH_USE_LIBAV=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONHASHSEED=random $(PYTHON_EXEC) -m pytest test/vad_tests.py | cat
	PYTHONPATH=. $(DYLD_ENV) TORCH_USE_LIBAV=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONHASHSEED=random $(PYTHON_EXEC) test/coqui_tts_test.py

certs: ## Generate SSL certificates for HTTPS/WSS.
	@mkdir -p certs
	openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365

clean: ## Clean up generated files and caches.
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	rm -f output_*.wav
	rm -rf ct2_models/*
	rm -rf speaker_voices/*
	@echo "Cleaned up build artifacts and generated files."

distclean: clean ## Clean up all generated files, caches, and downloaded models.
	@echo "--- Performing a deep clean (removing all downloaded models) ---"
	rm -rf backend/tts/piper_models/*.onnx
	rm -rf backend/tts/piper_models/*.json
	rm -rf $(VENV_NAME)
	@echo "Deep clean complete. You may need to run 'make install' again."

help: ## Display this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
