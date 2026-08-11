EXCLUDE_NEWER := 2026-04-30
# The ROCm 7.2 wheels on the PyTorch index are newer than EXCLUDE_NEWER (torch and
# torchvision were uploaded on 2026-07-08), so the ROCm lock needs its own date.
EXCLUDE_NEWER_ROCM := 2026-07-31
# `uv run` ignores index directives inside a --with-requirements file, so the index has to
# be passed on the command line there as well.
ROCM_INDEX_URL := https://download.pytorch.org/whl/rocm7.2
COCO_ROOT := "/datasets/coco"
VISDRONE_ROOT := /datasets/visdrone
VISDRONE_COCO_ROOT := /datasets/visdrone_coco_remapped
# The benchmark matches ground truth against predictions by class id, so it has to be
# pointed at the COCO-remapped copy of the dataset written by `make remap-visdrone`, not
# at the original VisDrone data config.
VISDRONE_DATA := $(VISDRONE_COCO_ROOT)/visdrone_coco_remapped.yaml
PINNED_PYTHON_VERSION := 3.13

.PHONY: lock
lock: lock-tensorrt lock-rocm

.PHONY: lock-tensorrt
lock-tensorrt:
	@echo "Locking requirements..."
	uv pip compile --exclude-newer $(EXCLUDE_NEWER) requirements/tensorrt.in -o requirements/tensorrt.txt

# lightly-train is omitted from the lock because it is installed editable from the local
# checkout, which would put a machine-specific absolute path into the lock file. The
# benchmark targets add it back with `uv run --with-editable ..`.
.PHONY: lock-rocm
lock-rocm:
	@echo "Locking ROCm requirements..."
	uv pip compile --exclude-newer $(EXCLUDE_NEWER_ROCM) \
		--python-version $(PINNED_PYTHON_VERSION) \
		--index-strategy unsafe-best-match \
		--no-emit-package lightly-train \
		requirements/rocm.in -o requirements/rocm.txt

.PHONY: remap-visdrone
remap-visdrone:
	uv run --python $(PINNED_PYTHON_VERSION) --no-project --with "pyyaml>=6" \
		remap_visdrone_to_coco.py --data $(VISDRONE_ROOT)/data.yaml \
		--out $(VISDRONE_COCO_ROOT) --overwrite

.PHONY: benchmark-tensorrt
benchmark-tensorrt:
	uv run --python $(PINNED_PYTHON_VERSION) --frozen --with-requirements requirements/tensorrt.txt run_tensorrt_benchmark.py --coco-root $(COCO_ROOT)

.PHONY: benchmark-torch-sahi
benchmark-torch-sahi:
	uv run --python $(PINNED_PYTHON_VERSION) --frozen --with-requirements requirements/tensorrt.txt run_torch_sahi_benchmark.py --data $(VISDRONE_DATA)

# --no-project keeps uv from layering the requirements on top of the repository's own
# environment, which holds a CUDA build of torch.
.PHONY: benchmark-torch-sahi-rocm
benchmark-torch-sahi-rocm:
	uv run --python $(PINNED_PYTHON_VERSION) --no-project \
		--extra-index-url $(ROCM_INDEX_URL) --index-strategy unsafe-best-match \
		--with-editable .. --with-requirements requirements/rocm.txt \
		run_torch_sahi_benchmark.py --data $(VISDRONE_DATA) \
		--out out/torch_sahi_benchmark_rocm

.PHONY: format
format:
	$(MAKE) -C .. format

.PHONY: format-check
format-check:
	$(MAKE) -C .. format-check

.PHONY: type-check
type-check:
	$(MAKE) -C .. type-check