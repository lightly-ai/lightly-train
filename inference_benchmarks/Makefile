EXCLUDE_NEWER := 2026-04-30
COCO_ROOT := "/datasets/coco"
PINNED_PYTHON_VERSION := 3.13

.PHONY: lock
lock:
	@echo "Locking requirements..."
	uv pip compile --exclude-newer $(EXCLUDE_NEWER) requirements/tensorrt.in -o requirements/tensorrt.txt

.PHONY: benchmark-tensorrt
benchmark-tensorrt:
	uv run --python $(PINNED_PYTHON_VERSION) --frozen --with-requirements requirements/tensorrt.txt run_tensorrt_benchmark.py --coco-root $(COCO_ROOT)

.PHONY: benchmark-tensorrt-resolutions
benchmark-tensorrt-resolutions:
	uv run --python $(PINNED_PYTHON_VERSION) --frozen --with-requirements requirements/tensorrt.txt run_tensorrt_resolution_benchmark.py --coco-root $(COCO_ROOT)

.PHONY: analyze-tensorrt-resolutions
analyze-tensorrt-resolutions:
	uv run --python $(PINNED_PYTHON_VERSION) --frozen --with-requirements requirements/tensorrt.txt analyze_tensorrt_resolution_pareto.py

.PHONY: format
format:
	$(MAKE) -C .. format

.PHONY: format-check
format-check:
	$(MAKE) -C .. format-check

.PHONY: type-check
type-check:
	$(MAKE) -C .. type-check
