# LightlyTrain Docker container

TODO

## Available images

List of currently available Docker base images:

- `amd64-cuda`
- More coming soon...

TODO(Malte, 06/2024): Rethink and rework the setup of supporting different base images
once we have multiple base images. Alternatives are e.g.:

1. Pass the base image type or directly the Dockerfile as argument to the makefile.
1. Put the Dockerfile, requirements and optionally makefile for each image type into
   a separate subdirectory.
1. Have docker multi-platform builds.

## Install

TODO

## Usage

First, start the docker container in interactive mode by using the -it flag. Furthermore,
you must mount the directories you want to use.

```
docker run -it --gpus=all --user $(id -u):$(id -g) -v /my_data_dir:/data -v /my_output_dir:/out lightly/train:latest
```

Then all the usual CLI commands are fully available. E.g. run

```
lightly-train train data="/data" out="/out" model="torchvision/convnext_small" method=dino
```

## Development

Note that there are different Dockerfiles and requirements file for each base image.

### Building images

Images can be built by calling the corresponding Makefile commands:

`make build-docker-IMAGE_TYPE` builds the image specified by the file `Dockerfile-IMAGE_TYPE`
and using the file `requirements-docker-IMAGE_TYPE.txt`

`make main-deploy-IMAGE_TYPE` builds the image and publishes it on Docker Hub.
