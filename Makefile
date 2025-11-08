.PHONY: install install-deps run test certs clean help

install: ## Install all Python dependencies, including PyTorch with MPS support.
	@echo "--- Uninstalling existing torch and torchaudio for a clean installation ---"
	pip uninstall -y torch torchaudio || true
	@echo "--- Installing all project dependencies from requirements.txt ---"
	pip install -r requirements.txt
	@echo "--- Downloading MT models (CTranslate2) ---"
	python backend/mt/convert_opus_mt_to_ct2.py Helsinki-NLP/opus-mt-en-sk
	python backend/mt/convert_opus_mt_to_ct2.py Helsinki-NLP/opus-mt-sk-en
	python backend/mt/convert_opus_mt_to_ct2.py Helsinki-NLP/opus-mt-en-cs
	@echo "--- Downloading Piper TTS models ---"
	python backend/tts/download_piper_models.py en_US-ryan-medium
	python backend/tts/download_piper_models.py sk_SK-lili-medium
	python backend/tts/download_piper_models.py cs_CZ-jirka-medium
	@echo "--- Installation complete. ---"

run: ## Run the FastAPI backend server.
	python app.py

test: ## Run the comprehensive backend test suite.
	python test/streaming_pipeline_tests.py

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
	rm -rf venv
	@echo "Deep clean complete. You may need to run 'make install' again."

help: ## Display this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
