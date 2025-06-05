# Import variables
include ./variables.mk

.PHONY: print-version
print-version:
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
		-v ./variables.mk:/home/lightly_train/docker/variables.mk \
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
