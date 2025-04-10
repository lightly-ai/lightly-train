TAG ?= $(shell git rev-parse HEAD)
IMAGE ?= train
DOCKER_BUILDKIT := 1
CONTAINER_REGISTRY := eu.gcr.io/boris-250909

VERSION := $(shell grep '__version__' ../src/lightly_train/__init__.py | sed -E 's/[^0-9.]//g')
VERSION_X := $(shell echo $(VERSION) | cut -d. -f1)
VERSION_XY := $(shell echo $(VERSION) | cut -d. -f1-2)

# print all versions
.PHONY: print-versions
print-versions:
	@echo "Version: $(VERSION)"



# Build the docker image using an amd64 and CUDA base image
.PHONY: build-docker-cuda  # Export image as local docker image
build-docker-amd64-cuda: NAMESPACE="lightly"
build-docker-amd64-cuda: OUTPUT="type=docker"
build-docker-amd64-cuda:
	DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker buildx build \
		--file Dockerfile-amd64-cuda \
		--tag $(NAMESPACE)/$(IMAGE):$(TAG) \
		--tag $(NAMESPACE)/$(IMAGE):latest \
		--target=runtime \
		--output=$(OUTPUT) \
		..

.PHONY: develop-deploy-amd64-cuda
develop-deploy-amd64-cuda: OUTPUT="type=registry"  # Push to registry
develop-deploy-amd64-cuda: NAMESPACE="lightly-develop"
develop-deploy-amd64-cuda:
	DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker buildx build \
		--file Dockerfile-amd64-cuda \
		--tag $(CONTAINER_REGISTRY)/$(NAMESPACE)/$(IMAGE):latest \
		--tag $(CONTAINER_REGISTRY)/$(NAMESPACE)/$(IMAGE):$(TAG) \
		--target=runtime \
		--output=$(OUTPUT) \
		..

.PHONY: main-deploy-amd64-cuda
main-deploy-amd64-cuda: OUTPUT="type=registry"  # Push to registry
main-deploy-amd64-cuda: NAMESPACE="lightly-main"
main-deploy-amd64-cuda:
	DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker buildx build \
		--file Dockerfile-amd64-cuda \
		--tag $(CONTAINER_REGISTRY)/$(NAMESPACE)/$(IMAGE):latest \
		--tag $(CONTAINER_REGISTRY)/$(NAMESPACE)/$(IMAGE):$(VERSION) \
		--tag $(CONTAINER_REGISTRY)/$(NAMESPACE)/$(IMAGE):$(TAG) \
		--target=runtime \
		--output=$(OUTPUT) \
		..

# DATE variable must be assigned with `:=` to avoid re-evaluation.
DATE := $(shell date +'%Y-%m-%d-%H-%M-%S')
LIGHTLY_TRAIN_OUT ?= $(PWD)/../out/docker/$(DATE)
LIGHTLY_TRAIN_DATA ?= $(PWD)/../lightly_train_docker_test_data

.PHONY: test
test:
	@echo "Generate images"
	mkdir -p $(LIGHTLY_TRAIN_DATA)
	python -c 'from PIL import Image; [Image.new("RGB", (250, 300)).save(f"$(LIGHTLY_TRAIN_DATA)/{i}.png") for i in range(5)]'
	@echo "Create output directory"
	mkdir -p $(LIGHTLY_TRAIN_OUT)
	chmod -R +rw $(LIGHTLY_TRAIN_OUT)
	docker run --rm --shm-size=1g --user $(shell id -u):$(shell id -g) \
		-v $(LIGHTLY_TRAIN_OUT):/out \
		-v $(LIGHTLY_TRAIN_DATA):/data \
		-v ./Makefile:/home/lightly_train/docker/Makefile \
		--gpus all \
		lightly/$(IMAGE):$(TAG) \
		make test-from-with-docker -C docker

test-from-with-docker:
	make test-pillow-simd-installed-within-docker
	make test-cli-from-within-docker
	

# This target is run from within the docker container
test-cli-from-within-docker:
	@echo "Test train"
	lightly-train train data=/data out=/out model="torchvision/convnext_small" epochs=2 batch_size=2 model_args.weights="IMAGENET1K_V1" devices=2
	test -f /out/checkpoints/last.ckpt
	test `grep -c "GPU available: True (cuda), used: True" /out/train.log` -gt 0
	@echo "Test embed"
	lightly-train embed data=/data out="/out/embeddings.csv" checkpoint="/out/checkpoints/last.ckpt" batch_size=2 format="csv"
	test `wc -l < /out/embeddings.csv` -eq 6
	@echo "Test export"
	lightly-train export out="/out/model.pth" checkpoint="/out/checkpoints/last.ckpt" part="model" format="torch_state_dict"
	test -f /out/model.pth

test-pillow-simd-installed-within-docker:
	@echo "Check that pillow-simd is installed"
	test `pip show pillow-simd | grep -c "Name: Pillow-SIMD"` -eq 1
	@echo "Check that pillow is not installed"
	test `pip show pillow 2>&1 | grep -c "WARNING: Package(s) not found: pillow"` -eq 1



### Release

# Takes the image from https://console.cloud.google.com/artifacts/docker/boris-250909/europe/eu.gcr.io/lightly-main%2Ftrain
# Pushes the image to dockerhub at lightly/train:latest.
# To check the published dockerhub images, go to https://hub.docker.com/repository/docker/lightly/train/general
.PHONY: release
release: SOURCE_IMAGE_NAMESPACE=lightly-main
release: SOURCE_IMAGE=$(CONTAINER_REGISTRY)/$(SOURCE_IMAGE_NAMESPACE)/$(IMAGE):$(TAG)
release: DOCKERHUB_NAMESPACE=lightly
release:
	docker buildx imagetools create --tag $(DOCKERHUB_NAMESPACE)/$(IMAGE):latest $(SOURCE_IMAGE)
	docker buildx imagetools create --tag $(DOCKERHUB_NAMESPACE)/$(IMAGE):$(VERSION) $(SOURCE_IMAGE)
	docker buildx imagetools create --tag $(DOCKERHUB_NAMESPACE)/$(IMAGE):$(VERSION_X) $(SOURCE_IMAGE)
	docker buildx imagetools create --tag $(DOCKERHUB_NAMESPACE)/$(IMAGE):$(VERSION_XY) $(SOURCE_IMAGE)

view-version:
	@echo "$(VERSION)"
