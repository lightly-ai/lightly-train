EXCLUDE_NEWER := 2026-04-30
COCO_ROOT := "/datasets/coco"

.PHONY: lock
lock:
	@echo "Locking requirements..."
	uv pip compile --exclude-newer $(EXCLUDE_NEWER) requirements/tensorrt.in -o requirements/tensorrt.txt

.PHONY: benchmark-tensorrt
benchmark-tensorrt:
	uv run --frozen --with-requirements requirements/tensorrt.txt run_tensorrt_benchmark.py --coco-root $(COCO_ROOT)

.PHONY: format
format:
	$(MAKE) -C .. format

.PHONY: format-check
format-check:
	$(MAKE) -C .. format-check

.PHONY: type-check
type-check:
	$(MAKE) -C .. type-check