EXCLUDE_NEWER := 2026-07-27
PINNED_PYTHON_VERSION := 3.10
BASE_PYTHON := /usr/bin/python3
VENV_PATH := /opt/venv
PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cu118

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPOSITORY_ROOT := $(abspath $(MAKEFILE_DIR)/../..)
REQUIREMENTS_DIRECTORY := .devcontainer/cpp-inference/requirements
REQUIREMENTS_IN := $(REQUIREMENTS_DIRECTORY)/requirements.in
REQUIREMENTS_LOCK := $(MAKEFILE_DIR)requirements/requirements.txt

# This container uses a dedicated, Python 3.10-specific requirements lock rather than
# the repository's universal uv.lock. Keep the dependency selection here in sync with
# the C++ inference environment.
UV_COMPILE_ARGS := --python-version $(PINNED_PYTHON_VERSION) --exclude-newer $(EXCLUDE_NEWER) \
	--group dev --group pinned-torch-maximal \
	--extra onnx --extra onnxruntime --extra onnxslim

# The +cu118 torch and torchvision wheels are published only on PyTorch's index. A
# compiled requirements file does not retain uv's project index configuration.
UV_SYNC_ARGS := --extra-index-url $(PYTORCH_INDEX_URL) --index-strategy unsafe-best-match

.PHONY: lock
lock:
	@echo "Locking requirements..."
	cd $(REPOSITORY_ROOT) && uv pip compile --directory . $(UV_COMPILE_ARGS) --emit-index-url \
		pyproject.toml $(REQUIREMENTS_IN) -o $(REQUIREMENTS_DIRECTORY)/requirements.txt

# Install the locked dependencies, then install the mounted repository editable. The
# local package is deliberately excluded from requirements.txt so this uses the
# current checkout and branch instead of a remote package revision.
.PHONY: install
install:
	uv venv $(VENV_PATH) --python $(BASE_PYTHON)
	uv pip sync --python $(VENV_PATH)/bin/python $(UV_SYNC_ARGS) $(REQUIREMENTS_LOCK)
	uv pip install --python $(VENV_PATH)/bin/python -e $(REPOSITORY_ROOT) --no-deps
