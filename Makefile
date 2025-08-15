### Cleaning

.PHONY: clean
clean: clean-build clean-pyc clean-out

# remove build artifacts
.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

# remove python file artifacts
.PHONY: clean-pyc
clean-pyc:
	find . -name '__pycache__' -exec rm -fr {} +

# remove hydra outputs
.PHONY: clean-out
clean-out:
	rm -fr outputs/
	rm -fr lightly_outputs/
	rm -fr lightning_logs/
	rm -fr lightly_epoch_*.ckpt
	rm -fr last.ckpt

### Formatting and type-checking

# run format and type checks and tests
.PHONY: all-checks
all-checks: static-checks test

# run format and type checks
.PHONY: static-checks
static-checks: format-check type-check

# Files to format with mdformat.
# This is needed to avoid formatting files in .venv. The mdformat command has an
# --exclude option but only on Python 3.13+.
MDFORMAT_FILES := .github docker docs src tests *.md

# run formatter
.PHONY: format
format: add-header
	# Format code
	ruff format .
	# Fix linting issues and sort imports
	ruff check --fix .
	# Format markdown files
	mdformat ${MDFORMAT_FILES}
	# Format code in markdown files
	pytest --update-examples docs/format_code.py::test_format_code_in_docs
	# Run pre-commit hooks
	pre-commit run --all-files

# run format check
.PHONY: format-check
format-check:
	# Check code formatting
	ruff format --check .
	# Check linting issues
	ruff check .
	# Check markdown formatting
	mdformat --check ${MDFORMAT_FILES}
	# Check code in markdown files
	pytest docs/format_code.py::test_format_check_code_in_docs
	# Run pre-commit hooks
	pre-commit run --all-files

# run type check
.PHONY: type-check
type-check:
	mypy src tests docs/format_code.py

# adding the license header to all files
.PHONY: add-header
add-header:
	licenseheaders -t dev_tools/licenseheader.tmpl -d src \
		-x src/lightly_train/_methods/dinov2/dinov2_loss.py \
		-x src/lightly_train/_methods/dinov2/dinov2_head.py \
		-x src/lightly_train/_methods/dinov2/utils.py \
		-x src/lightly_train/_modules/teachers/dinov2 \
		-x src/lightly_train/_lightning_rank_zero.py \
		-x src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/mask_loss.py \
		-x src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scale_block.py \
		-x src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scheduler.py \
		-x src/lightly_train/_models/dinov3/dinov3_src \
		-E py
	licenseheaders -t dev_tools/licenseheader.tmpl -d tests

	# Apply the Apache 2.0 license header to DINOv2-derived files
	licenseheaders -t dev_tools/dinov2_licenseheader.tmpl \
		-d src/lightly_train/_models/dinov2_vit/dinov2_vit_src \
		-E py
	
	licenseheaders -t dev_tools/dinov2_licenseheader.tmpl \
		-f src/lightly_train/_methods/dinov2/dinov2_loss.py \
		src/lightly_train/_methods/dinov2/dinov2_head.py \
		src/lightly_train/_methods/dinov2/utils.py \
		-E py

	# Apply the Apache 2.0 license header to PyTorch Lighting derived files
	licenseheaders -t dev_tools/pytorch_lightning_licenseheader.tmpl \
		-f src/lightly_train/_lightning_rank_zero.py

	# Apply the MIT license header to the EoMT derived files
	licenseheaders -t dev_tools/eomt_licenseheader.tmpl \
		-f src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scheduler.py \
		-E py
	
	# Apply the DINOv3 license header to the DINOv3 derived files
	licenseheaders -t dev_tools/dinov3_licenseheader.tmpl \
		-d src/lightly_train/_models/dinov3/dinov3_src \
		-E py


### Testing

# run tests
.PHONY: test
test:
	pytest tests

.PHONY: test-ci
test-ci:
	pytest tests -v


### Virtual Environment

.PHONY: install-uv
install-uv:
	curl -LsSf https://astral.sh/uv/0.5.4/install.sh | sh


.PHONY: reset-venv
reset-venv:
	deactivate || true
	rm -rf .venv
	uv venv .venv


### Dependencies

# When running these commands locally, it is recommended to first reset the environment
# with: `make reset-venv && source .venv/bin/activate`
# Otherwise old dependencies might linger around.

# Set EDITABLE to -e to install the package in editable mode outside of CI. This is
# useful for local development.
ifdef CI
EDITABLE :=
else
EDITABLE := -e
endif

# RFDETR and ONNXRuntime is not compatible with Python<3.9. Therefore we exclude it from the
# default extras.
EXTRAS_PY38 := [dev,mlflow,notebook,onnx,super-gradients,tensorboard,timm,ultralytics,wandb]

# SuperGradients is not compatible with Python>=3.10. It is also not easy to install
# on MacOS. Therefore we exclude it from the default extras.
# RFDETR has installation issues because of onnxsim dependency on CI with Python 3.12.
# Onnx dependencies in RFDETR should become optional in RFDETR >1.1.0.
EXTRAS_PY312 := [dev,mlflow,notebook,onnx,onnxruntime,tensorboard,timm,ultralytics,wandb]

# RF-DETR is not always installable for Python>=3.12, therefore we remove it from the
# default development dependencies. And SuperGradients is not compatible with
# Python>=3.10, therefore we also remove it from the default development dependencies.
EXTRAS_DEV := [dev,mlflow,notebook,onnx,onnxruntime,tensorboard,timm,ultralytics,wandb]

# Exclude ultralytics from docker extras as it has an AGPL license and we should not
# distribute it with the docker image.
DOCKER_EXTRAS := --extra mlflow --extra tensorboard --extra timm --extra wandb --extra rfdetr

# Date until which dependencies installed with --exclude-newer must have been released.
# Dependencies released after this date are ignored.
EXCLUDE_NEWER_DATE := "2025-06-08"

# Pinned versions for Torch and TorchVision to avoid issues with the CUDA/driver version
# on the CI machine. These versions are compatible with CUDA 11.4 and Python 3.8.
# They are the latest versions available before 2024-08-28.
# Torch 2.5+ is no longer compatible with Python 3.8.
# Be careful when making changes on the CI machine as other repositories also depend on
# the installed CUDA/driver versions.
#
# The CI versions are pinned to specific URLs as specifying them as simple version string
# (e.g. "torch==2.4.0") with the --index-url or --extra-index-url options from UV leads
# down a rabbit hole of dependency resolution issues.
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
ifdef CI
PINNED_TORCH_VERSION_PY38 := "torch@https://download.pytorch.org/whl/cu118/torch-2.4.0%2Bcu118-cp38-cp38-linux_x86_64.whl"
PINNED_TORCH_VERSION_PY312 := "torch@https://download.pytorch.org/whl/cu118/torch-2.4.0%2Bcu118-cp312-cp312-linux_x86_64.whl"
PINNED_TORCHVISION_VERSION_PY38 := "torchvision@https://download.pytorch.org/whl/cu118/torchvision-0.19.0%2Bcu118-cp38-cp38-linux_x86_64.whl"
PINNED_TORCHVISION_VERSION_PY312 := "torchvision@https://download.pytorch.org/whl/cu118/torchvision-0.19.0%2Bcu118-cp312-cp312-linux_x86_64.whl"
MINIMAL_TORCH_VERSION_PY38 := "torch@https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp38-cp38-linux_x86_64.whl"
MINIMAL_TORCHVISION_VERSION_PY38 := "torchvision@https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp38-cp38-linux_x86_64.whl"
endif
else
PINNED_TORCH_VERSION_PY38 := "torch==2.4.0"
PINNED_TORCH_VERSION_PY312 := "torch==2.4.0"
PINNED_TORCHVISION_VERSION_PY38 := "torchvision==0.19.0"
PINNED_TORCHVISION_VERSION_PY312 := "torchvision==0.19.0"
MINIMAL_TORCH_VERSION_PY38 := "torch==2.1.0"
MINIMAL_TORCHVISION_VERSION_PY38 := "torchvision==0.16.0"
endif


# Install ffmpeg on Ubuntu.
. PHONY: install-ffmpeg-ubuntu
install-ffmpeg-ubuntu:
	sudo apt-get install ffmpeg=7:4.2.7-0ubuntu0.1

# Install package for local development.
.PHONY: install-dev
install-dev:
	uv pip install ${EDITABLE} ".${EXTRAS_DEV}"
	pre-commit install

# Install package with minimal dependencies.
#
# This command is split into multiple steps:
# 1. Install the dev dependencies to be able to run tests. We don't want to use
#    the minimal versions for these dependencies.
# 2. Then we reinstall the package with minimal dependencies.
#
# Explanation of flags:
# --exclude-newer: We don't want to install dependencies released after that date to
#   keep CI stable.
# --resolution=lowest-direct: Only install minimal versions for direct dependencies.
#   Transitive dependencies will use the latest compatible version.
# 	Using --resolution=lowest would also download the latest versions for transitive
#   dependencies which is not a realistic scenario and results in some extremely old
#   dependencies being installed.
# --reinstall: Reinstall dependencies to make sure they satisfy the constraints.
.PHONY: install-minimal
install-minimal:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} ${EDITABLE} ".[dev]"
	uv pip install --resolution=lowest-direct --exclude-newer ${EXCLUDE_NEWER_DATE} \
		--reinstall ${EDITABLE} "." --requirement pyproject.toml \
		${MINIMAL_TORCH_VERSION_PY38} ${MINIMAL_TORCHVISION_VERSION_PY38}

# Install package with minimal dependencies including extras.
# See install-minimal for more information.
.PHONY: install-minimal-extras
install-minimal-extras:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} ${EDITABLE} ".[dev]"
	uv pip install --resolution=lowest-direct --exclude-newer ${EXCLUDE_NEWER_DATE} \
		--reinstall ${EDITABLE} ".${EXTRAS_PY38}" --requirement pyproject.toml \
		${MINIMAL_TORCH_VERSION_PY38} ${MINIMAL_TORCHVISION_VERSION_PY38}

# Install package for Python 3.8 with dependencies pinned to the latest compatible
# version available at EXCLUDE_NEWER_DATE. This keeps CI stable if new versions of
# dependencies are released.
#
# We have to differentiate between Python versions because SuperGradients is not
# compatible with Python>=3.10, while RFDETR is not compatible with Python<3.9 and also
# installing it on Python>=3.12 can produce issues with cmake.
# For Python 3.8 we install the package with SuperGradients and without RFDETR.
# Torch and TorchVision are pinned to specific versions to avoid issues with the
# CUDA/driver version on the CI machine.
.PHONY: install-pinned-3.8
install-pinned-3.8:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} ".${EXTRAS_PY38}" --requirement pyproject.toml \
		${PINNED_TORCH_VERSION_PY38} ${PINNED_TORCHVISION_VERSION_PY38}

# Install package for Python 3.12 with dependencies pinned to the latest compatible
# version available at EXCLUDE_NEWER_DATE.
#
# For Python 3.12 we install the package with RFDETR and without SuperGradients.
# See install-pinned-3.8 for more information.
.PHONY: install-pinned-3.12
install-pinned-3.12:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} ".${EXTRAS_PY312}" --requirement pyproject.toml \
		${PINNED_TORCH_VERSION_PY312} ${PINNED_TORCHVISION_VERSION_PY312}

# Install package with the latest dependencies for Python 3.8.
.PHONY: install-latest-3.8
install-latest-3.8:
	uv pip install --upgrade --reinstall ${EDITABLE} ".${EXTRAS_PY38}"

# Install package with the latest dependencies for Python 3.12.
.PHONY: install-latest-3.12
install-latest-3.12:
	uv pip install --upgrade --reinstall ${EDITABLE} ".${EXTRAS_PY312}"

# Install package with same dependencies as for Python 3.12
.PHONY: install-latest-3.13
install-latest-3.13: install-latest-3.12

# Install package for building docs.
.PHONY: install-docs
install-docs:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${EDITABLE} ".${EXTRAS_PY312}" --requirement pyproject.toml

# Install package dependencies in Docker image.
# Uninstall opencv-python and opencv-python-headless because they are both installed by rfdetr
# but only one of them is needed.
# Uninstall pillow because we want to install pillow-simd instead.
.PHONY: install-docker-dependencies
install-docker-dependencies:
	uv pip install -v --exclude-newer ${EXCLUDE_NEWER_DATE} ${DOCKER_EXTRAS} --requirement pyproject.toml
	uv pip uninstall opencv-python opencv-python-headless
	uv pip install opencv-python-headless
	uv pip uninstall pillow
	C="cc -mavx2" uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} --upgrade --force-reinstall pillow-simd

# Install package in Docker image.
# This requires `install-docker-dependencies` to be run first. We don't add this command
# as a dependency to not run it multiple times accidentally.
.PHONY: install-docker
install-docker:
	uv pip install -v --no-deps .

# Install dependencies for building and publishing the package.
.PHONY: install-dist
install-dist:
	uv pip install --exclude-newer ${EXCLUDE_NEWER_DATE} wheel twine build 

### Building source and wheel package for publishing to pypi
.PHONY: dist
dist: clean
	python -m build
	ls -l dist


### Downloads

# Download the models used in the docker image.
# Models are saved to LIGHTLY_TRAIN_CACHE_DIR location.
.PHONY: download-docker-models
download-docker-models:
	curl -o "${LIGHTLY_TRAIN_CACHE_DIR}/weights/dinov2_vitb14_pretrain.pth" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
