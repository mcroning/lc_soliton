.PHONY: test gpu-test install-dev clean

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest -q

gpu-test:
	pytest -q tests/test_gpu_available.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf build dist .pytest_cache
