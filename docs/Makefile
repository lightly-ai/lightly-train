### Documentation

VERSION ?= $(shell grep '__version__' ../src/lightly_train/__init__.py | sed -E 's/[^0-9.]//g')
BUILD_DIR := build
# The build directory for the current version. We create a new directory for each
# version to simplify hosting the documentation for multiple versions. This follows
# the pattern used by PyTorch: https://pytorch.org/docs/versions.html
BUILD_VERSION_DIR := ${BUILD_DIR}/${VERSION}
BUILD_STABLE_DIR := ${BUILD_DIR}/stable
SOURCE_DIR := source


# Build docs for the current version.
.PHONY: docs
docs:
	@echo "📚 Building v${VERSION} documentation..."
	uv run --frozen python prebuild.py --source-dir ${SOURCE_DIR}
	uv run --frozen sphinx-build -b html --fail-on-warning --keep-going ${SOURCE_DIR} ${BUILD_VERSION_DIR}
	uv run --frozen python build.py --build-dir ${BUILD_DIR}
	@echo "✅ Documentation built successfully!"

# Build docs for the stable version. Assumes that the current version is the stable
# version.
.PHONY: docs-stable
docs-stable:
	@echo "🗿 Building stable documentation..."
	uv run --frozen python prebuild.py --source-dir ${SOURCE_DIR}
	uv run --frozen sphinx-build -b html --fail-on-warning --keep-going ${SOURCE_DIR} ${BUILD_STABLE_DIR}
	uv run --frozen python build.py --build-dir ${BUILD_DIR}
	@echo "✅ Documentation built successfully!"

# Serve the documentation on localhost.
.PHONY: serve
serve:
	uv run --frozen python -m http.server 1234 --directory ${BUILD_DIR}

.PHONY: clean
clean:
	rm -rf ${BUILD_DIR}

.PHONY: format
format:
	make -C .. format

.PHONY: format-check
format-check:
	make -C .. format-check