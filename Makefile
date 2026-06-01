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
	uv run --frozen ruff format .
	# Fix linting issues and sort imports
	uv run --frozen ruff check --fix .
	# Format markdown files
	uv run --frozen mdformat ${MDFORMAT_FILES}
	# Format code in markdown files
	uv run --frozen pytest --update-examples docs/format_code.py::test_format_code_in_docs
	# Run pre-commit hooks
	uv run --frozen pre-commit run --all-files

# run format check
.PHONY: format-check
format-check:
	# Check code formatting
	uv run --frozen ruff format --check .
	# Check linting issues
	uv run --frozen ruff check .
	# Check markdown formatting
	uv run --frozen mdformat --check ${MDFORMAT_FILES}
	# Check code in markdown files
	uv run --frozen pytest docs/format_code.py::test_format_check_code_in_docs
	# Run pre-commit hooks
	uv run --frozen pre-commit run --all-files

# run type check
.PHONY: type-check
type-check:
	uv run --frozen mypy src tests docs/format_code.py

# adding the license header to all files
.PHONY: add-header
add-header:
	uv run --frozen licenseheaders -t dev_tools/licenseheader.tmpl -d src \
		-x src/lightly_train/_methods/dinov2/dinov2_loss.py \
		-x src/lightly_train/_methods/dinov2/dinov2_head.py \
		-x src/lightly_train/_methods/dinov2/utils.py \
		-x src/lightly_train/_modules/teachers/dinov2 \
		-x src/lightly_train/_lightning_rank_zero.py \
		-x src/lightly_train/_task_models/dinov2_eomt_panoptic_segmentation/mask_loss.py \
		-x src/lightly_train/_task_models/dinov2_eomt_panoptic_segmentation/scale_block.py \
		-x src/lightly_train/_task_models/dinov2_eomt_panoptic_segmentation/scheduler.py \
		-x src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/mask_loss.py \
		-x src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scale_block.py \
		-x src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scheduler.py \
		-x src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/mask_loss.py \
		-x src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/scale_block.py \
		-x src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/scheduler.py \
		-x src/lightly_train/_task_models/dinov3_eomt_panoptic_segmentation/mask_loss.py \
		-x src/lightly_train/_task_models/dinov3_eomt_panoptic_segmentation/scale_block.py \
		-x src/lightly_train/_task_models/dinov3_eomt_panoptic_segmentation/scheduler.py \
		-x src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/mask_loss.py \
		-x src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/scale_block.py \
		-x src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/scheduler.py \
		-x src/lightly_train/_models/dinov3/dinov3_src \
		-x src/lightly_train/_task_models/object_detection_components \
		-x src/lightly_train/_task_models/picodet_object_detection/csp_pan.py \
		-x src/lightly_train/_task_models/picodet_object_detection/esnet.py \
		-x src/lightly_train/_task_models/picodet_object_detection/losses.py \
		-x src/lightly_train/_task_models/picodet_object_detection/pico_head.py \
		-E py
	uv run --frozen licenseheaders -t dev_tools/licenseheader.tmpl -d tests

	# Apply the Apache 2.0 license header to DINOv2-derived files
	uv run --frozen licenseheaders -t dev_tools/dinov2_licenseheader.tmpl \
		-d src/lightly_train/_models/dinov2_vit/dinov2_vit_src \
		-E py
	
	uv run --frozen licenseheaders -t dev_tools/dinov2_licenseheader.tmpl \
		-f src/lightly_train/_methods/dinov2/dinov2_loss.py \
		src/lightly_train/_methods/dinov2/dinov2_head.py \
		src/lightly_train/_methods/dinov2/utils.py \
		-E py

	# Apply the Apache 2.0 license header to PyTorch Lighting derived files
	uv run --frozen licenseheaders -t dev_tools/pytorch_lightning_licenseheader.tmpl \
		-f src/lightly_train/_lightning_rank_zero.py

	# Apply the Apache 2.0 license header to RT-DETR derived files
	uv run --frozen licenseheaders -t dev_tools/rtdetr_licenseheader.tmpl \
		-d src/lightly_train/_task_models/object_detection_components/ \
		-x src/lightly_train/_task_models/object_detection_components/tiling_utils.py \
		   src/lightly_train/_task_models/object_detection_components/dfine_decoder.py \
		   src/lightly_train/_task_models/object_detection_components/dfine_utils.py \
		   src/lightly_train/_task_models/object_detection_components/dfine_criterion.py \
		-E py

	# Apply Lightly's header to tiling_utils.py
	uv run --frozen licenseheaders -t dev_tools/licenseheader.tmpl \
		-f src/lightly_train/_task_models/object_detection_components/tiling_utils.py \
		-E py

	# Apply the Apache 2.0 license header to D-FINE derived files
	uv run --frozen licenseheaders -t dev_tools/dfine_licenseheader.tmpl \
		-f src/lightly_train/_task_models/object_detection_components/dfine_decoder.py \
		src/lightly_train/_task_models/object_detection_components/dfine_utils.py \
		src/lightly_train/_task_models/object_detection_components/dfine_criterion.py \
		-E py

	# Apply the PicoDet license header to PicoDet-derived files
	uv run --frozen licenseheaders -t dev_tools/picodet_licenseheader.tmpl \
		-f src/lightly_train/_task_models/picodet_object_detection/csp_pan.py \
		src/lightly_train/_task_models/picodet_object_detection/esnet.py \
		src/lightly_train/_task_models/picodet_object_detection/losses.py \
		src/lightly_train/_task_models/picodet_object_detection/pico_head.py \
		-E py

	# Apply the Apache 2.0 license header to DEIMv2 derived files
	uv run --frozen licenseheaders -t dev_tools/deimv2_licenseheader.tmpl \
		-f src/lightly_train/_task_models/dinov3_ltdetr_object_detection/dinov3_vit_wrapper.py \
		src/lightly_train/_task_models/object_detection_components/flat_cosine.py \
		-E py

	# Apply the MIT license header to the EoMT derived files
	uv run --frozen licenseheaders -t dev_tools/eomt_licenseheader.tmpl \
		-f src/lightly_train/_task_models/dinov2_eomt_panoptic_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov2_eomt_panoptic_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov2_eomt_panoptic_segmentation/scheduler.py \
		src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov2_eomt_semantic_segmentation/scheduler.py \
		src/lightly_train/_task_models/dinov2_eomt_instance_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov2_eomt_instance_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov2_eomt_instance_segmentation/scheduler.py \
		src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov3_eomt_instance_segmentation/scheduler.py \
		src/lightly_train/_task_models/dinov3_eomt_panoptic_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov3_eomt_panoptic_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov3_eomt_panoptic_segmentation/scheduler.py \
		src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/mask_loss.py \
		src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/scale_block.py \
		src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/scheduler.py \
		-E py
	
	# Apply the DINOv3 license header to the DINOv3 derived files
	uv run --frozen licenseheaders -t dev_tools/dinov3_licenseheader.tmpl \
		-d src/lightly_train/_models/dinov3/dinov3_src \
		-E py


### Testing

# run tests
.PHONY: test
test:
	uv run --frozen pytest tests

.PHONY: test-ci-minimal
test-ci-minimal:
	uv run --frozen --group pinned-torch-minimal pytest tests -v --durations=20

.PHONY: test-ci-maximal
test-ci-maximal:
	uv run --frozen --group pinned-torch-maximal pytest tests -v --durations=20


### Virtual Environment

.PHONY: install-uv
install-uv:
	curl -LsSf https://astral.sh/uv/0.10.0/install.sh | sh


.PHONY: remove-venv
remove-venv:
	deactivate || true
	rm -rf .venv


### Dependencies

# Set NO_EDITABLE to install the package in non-editable mode outside of CI. This is
# useful for local development.
ifdef CI
NO_EDITABLE := --no-editable
else
NO_EDITABLE :=
endif

# Extras list to uv command line arguments. E.g., [a,b] becomes --extra a --extra b.
comma := ,
to_uv_extras = --extra $(subst $(comma), --extra ,$(subst ],,$(subst [,,$(1))))

# Min and max Python versions for testing.
MINIMAL_PYTHON_VERSION := 3.8
MAXIMAL_PYTHON_VERSION := 3.13

# Notebook is excluded from the Python 3.8 extras because its dependency pywinpty is not
# compatible with uv 0.10.0 on Python 3.8. It fails with:
#   × Failed to download and build `pywinpty==2.0.14`
#   ├─▶ Failed to parse:
#   │   `D:\a\_temp\setup-uv-cache\sdists-v9\pypi\pywinpty\2.0.14\pQ3SKTY_3WgM9I6D1MxET\src\pyproject.toml`
#   ╰─▶ TOML parse error at line 1, column 1
#         |
#       1 | [project]
#         | ^^^^^^^^^
#       `pyproject.toml` is using the `[project]` table, but the required
#       `project.version` field is neither set nor present in the
#       `project.dynamic` list
# 
# This problem is fixed in pywinpty>2.0.14 but these versions are not compatible with
# Python 3.8.
#
# SuperGradients is excluded as it is outdated and causes issues in CI.
#
# RFDETR and ONNXRuntime are not compatible with Python<3.9. Therefore we exclude it 
# from the default extras.
EXTRAS_PY38 := [dicom,mlflow,onnx,tensorboard,timm,ultralytics,wandb]

# SuperGradients is excluded as it is not compatible with Python>=3.10.
EXTRAS_PY313 := [dicom,mlflow,notebook,onnx,onnxruntime,onnxslim,rfdetr,tensorboard,timm,ultralytics,wandb]

# SuperGradients is excluded as it is not compatible with Python>=3.10.
EXTRAS_DEV := [dicom,mlflow,notebook,onnx,onnxruntime,onnxslim,rfdetr,tensorboard,timm,ultralytics,wandb]

# Exclude ultralytics from docker extras as it has an AGPL license and we should not
# distribute it with the docker image.
DOCKER_EXTRAS := [mlflow,tensorboard,timm,wandb,rfdetr]

# Date until which dependencies installed with --exclude-newer must have been released.
# Dependencies released after this date are ignored.
EXCLUDE_NEWER_DATE := "2026-05-18"

export LIGHTLY_TRAIN_EVENTS_DISABLED := "1"
export LIGHTLY_TRAIN_POSTHOG_KEY := ""

# Install ffmpeg on Ubuntu.
.PHONY: install-ffmpeg-ubuntu
install-ffmpeg-ubuntu:
	sudo apt-get install ffmpeg=7:4.2.7-0ubuntu0.1

.PHONY: lock
lock:
	uv lock --exclude-newer ${EXCLUDE_NEWER_DATE}

# Install package for local development. Don't resolve, use lock file.
.PHONY: install-dev
install-dev:
	uv sync --frozen ${NO_EDITABLE} --group dev $(call to_uv_extras,$(EXTRAS_DEV))
	uv run --frozen pre-commit install

# Install package with minimal dependencies and latest development dependencies.
#
# Explanation of flags:
# --exclude-newer: We don't want to install dependencies released after that date to
#   keep CI stable.
# --resolution=lowest-direct: Only install minimal versions for direct dependencies.
#   Transitive dependencies will use the latest compatible version.
# 	Using --resolution=lowest would also download the latest versions for transitive
#   dependencies which is not a realistic scenario and results in some extremely old
#   dependencies being installed.
.PHONY: install-minimal
install-minimal:
	uv sync --python=${MINIMAL_PYTHON_VERSION} --resolution=lowest-direct \
		--exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --group dev \
		--group minimal-torch --upgrade-group dev

# Install package with minimal dependencies and extras, but latest versions of
# development packages.
.PHONY: install-minimal-extras
install-minimal-extras:
	uv sync --python=${MINIMAL_PYTHON_VERSION} --resolution=lowest-direct \
		--exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --group dev \
		--group minimal-torch $(call to_uv_extras,$(EXTRAS_PY38)) \
		--upgrade-group dev

# Install package for Python 3.8 with dependencies pinned to the latest compatible
# version available at EXCLUDE_NEWER_DATE. This keeps CI stable if new versions of
# dependencies are released.
# 
# SuperGradients is excluded as it is outdated and causes issues in CI.
# Torch and TorchVision are pinned to specific versions to avoid issues with the
# CUDA/driver version on the CI machine.
.PHONY: install-pinned-3.8
install-pinned-3.8:
	uv sync --frozen --python=${MINIMAL_PYTHON_VERSION} \
		--exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --upgrade-group dev \
		--group pinned-torch-minimal $(call to_uv_extras,$(EXTRAS_PY38))

# Install package for Python 3.13 with dependencies pinned to the latest compatible
# version available at EXCLUDE_NEWER_DATE.
#
# See install-pinned-3.8 for more information.
.PHONY: install-pinned-3.13
install-pinned-3.13:
	uv sync --frozen --python=${MAXIMAL_PYTHON_VERSION} \
		--exclude-newer ${EXCLUDE_NEWER_DATE} ${NO_EDITABLE} --upgrade-group dev \
		--group pinned-torch-maximal $(call to_uv_extras,$(EXTRAS_PY313))

# Install package with the latest dependencies for Python 3.8. The --upgrade flag
# ensures that the lockfile is ignored.
.PHONY: install-latest-3.8
install-latest-3.8:
	uv sync --python=${MINIMAL_PYTHON_VERSION} --upgrade --reinstall ${NO_EDITABLE} \
		--group dev $(call to_uv_extras,$(EXTRAS_PY38))

# Install package with the latest dependencies for Python 3.13. The --upgrade flag
# ensures that the lockfile is ignored.
.PHONY: install-latest-3.13
install-latest-3.13:
	uv sync --python=${MAXIMAL_PYTHON_VERSION} --upgrade --reinstall ${NO_EDITABLE} \
		--group dev $(call to_uv_extras,$(EXTRAS_PY313))

# Install package for building docs.
.PHONY: install-docs
install-docs:
	uv sync --frozen --python=${MAXIMAL_PYTHON_VERSION} \
		--exclude-newer ${EXCLUDE_NEWER_DATE} --reinstall ${NO_EDITABLE} \
		--group dev $(call to_uv_extras,$(EXTRAS_PY313))

# Install package dependencies in Docker image.
# Uninstall opencv-python and opencv-python-headless because they are both installed by rfdetr
# but only one of them is needed.
# Uninstall pillow because we want to install pillow-simd instead.
.PHONY: install-docker-dependencies
install-docker-dependencies:
	uv pip install -v --exclude-newer ${EXCLUDE_NEWER_DATE} $(call to_uv_extras,$(DOCKER_EXTRAS)) --requirement pyproject.toml
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
	uv sync --frozen --python=${MAXIMAL_PYTHON_VERSION} \
		--exclude-newer ${EXCLUDE_NEWER_DATE} --only-group dist

### Building source and wheel package for publishing to pypi
.PHONY: dist
dist: clean
	uv run --frozen python -m build
	ls -l dist


### Downloads

# Download the models used in the docker image.
# Models are saved to LIGHTLY_TRAIN_CACHE_DIR location.
.PHONY: download-docker-models
download-docker-models:
	curl -o "${LIGHTLY_TRAIN_CACHE_DIR}/weights/dinov3_vitb16_lvd1689m.pth" https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitb16_lvd1689m.pth
