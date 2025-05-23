# LightlyTrain Docker Image

Documentation on how to use the docker images: https://docs.lightly.ai/train/stable/docker.html

## Available Images

List of currently available Docker base images:

- `amd64-cuda`
- More coming soon...

TODO(Malte, 06/2024): Rethink and rework the setup of supporting different base images
once we have multiple base images. Alternatives are e.g.:

1. Pass the base image type or directly the Dockerfile as argument to the makefile.
1. Put the Dockerfile, requirements and optionally makefile for each image type into
   a separate subdirectory.
1. Have docker multi-platform builds.

## Development

### Building Images

Images are built by calling the corresponding [Makefile](./Makefile) command:

- `make build-docker-IMAGE_TYPE` builds the image specified by the file `Dockerfile-IMAGE_TYPE`

### Testing Images

Run tests with `make test`. The docker image must already be built.

Attention! This requires that a Python environment is activated. It will also create
some images locally (outside of the docker container) for testing.
