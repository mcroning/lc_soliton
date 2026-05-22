.PHONY: test gpu-test validate install-dev clean

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest -q

gpu-test:
	pytest -q tests/test_gpu_available.py

validate:
	python validation/validate_tiny_static.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf build dist .pytest_cache
