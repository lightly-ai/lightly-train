EXCLUDE_NEWER := 2026-07-27
PINNED_PYTHON_VERSION := 3.10

.PHONY: lock
lock:
	@echo "Locking requirements..."
	uv pip compile --directory ../.. --python-version $(PINNED_PYTHON_VERSION) --exclude-newer $(EXCLUDE_NEWER) \
		--group dev --group pinned-torch-maximal \
		--extra onnx --extra onnxruntime --extra onnxslim \
		--emit-index-url \
		pyproject.toml .devcontainer/cpp-inference/requirements/requirements.in -o .devcontainer/cpp-inference/requirements/requirements.txt
