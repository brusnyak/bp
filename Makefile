.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	python backend/main.py

test:
	python test/backend_test.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	rm -f output_*.wav
