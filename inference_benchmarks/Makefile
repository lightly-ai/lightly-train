EXCLUDE_NEWER := 2026-04-30

.PHONY: lock
lock:
	@echo "Locking requirements..."
	uv pip compile --exclude-newer $(EXCLUDE_NEWER) requirements/tensorrt.in -o requirements/tensorrt.txt