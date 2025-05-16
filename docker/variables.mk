# This file must only contain variables and no targets.

# Directory of this file
THIS_FILE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

TAG ?= $(shell git -C $(THIS_FILE_DIR) rev-parse HEAD)
IMAGE ?= train
DOCKER_BUILDKIT := 1

VERSION := $(shell grep '__version__' $(THIS_FILE_DIR)/../src/lightly_train/__init__.py | sed -E 's/[^0-9.]//g')
VERSION_X := $(shell echo $(VERSION) | cut -d. -f1)
VERSION_XY := $(shell echo $(VERSION) | cut -d. -f1-2)
