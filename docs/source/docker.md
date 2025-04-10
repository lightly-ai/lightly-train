(docker)=

# Docker

Lightly**Train** is available as a Docker image on [Docker Hub](https://hub.docker.com/r/lightly/train)
for containerized deployment.

(docker-installation)=

## Installation

Install Docker with NVIDIA GPU support ([docs](https://docs.lightly.ai/docs/lightly-worker-installation-guide#install-docker-optionally-with-gpu-support)).
Then, pull the latest version of the Lightly**Train** Docker image from Docker Hub:

```bash
docker pull lightly/train:latest
```

You can verify that the image is working correctly by running the following command:

```bash
docker run --rm --gpus=all lightly/train:latest lightly-train --help
```

This should print the Lightly**Train** help message.

(docker-usage)=

## Usage

Start a Lightly**Train** Docker container in interactive mode:

```bash
docker run -it --rm --gpus=all --shm-size=4gb --user $(id -u):$(id -g) -v /my_output_dir:/out -v /my_data_dir:/data lightly/train:latest
```

Flags:

- `-it`: Starts the container in interactive mode.
- `--rm`: Removes the container after it has been stopped.
- `--gpus=all`: Enables GPU support.
- `--shm-size=4gb`: Sets the shared memory size to 4 GB. Increase this for large datasets.
- `--user $(id -u):$(id -g)`: Run the container with the same user as the host. This
  makes sure that all files created by the container (checkpoints, logs, etc.) have
  the same permissions as the user running the container.
- `-v /my_output_dir:/out`: Mounts the host directory `/my_output_dir` to the container
  directory `/out`. All files created by the container (checkpoints, logs, etc.) will
  be saved in this directory.
- `-v /my_data_dir:/data`: Mounts the host directory `/my_data_dir` to the container
  directory `/data`. This directory must contain your training data. See the
  [Data](#train-data) docs for more information on how to structure your data.

Once the container is running, you can run Lightly**Train** commands inside the container
as if you had installed it locally. The only difference is that paths must be specified
relative to the mounted directories `/out` and `/data`. For example, to train a model,
run the following command inside the container:

````{tab} Command Line
```bash
lightly-train train out="/out/my_experiment" data="/data" model="torchvision/resnet50"
```
````
