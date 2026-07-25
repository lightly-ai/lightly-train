# LightlyTrain with MIGraphX

This example provides a container with AMD ROCm, MIGraphX, and the ONNX export
dependencies of LightlyTrain. It avoids installing ROCm or MIGraphX in the host
userspace, and installs LightlyTrain from the local repository source used as
the Docker build context.

The host still needs a supported AMD GPU driver and access to `/dev/kfd` and
`/dev/dri`. MIGraphX is installed from AMD's ROCm apt repository because it is
not available as a standalone PyPI package.

## Build

From the repository root:

```bash
docker build --tag lightly-train-migraphx -f examples/migraphx/Dockerfile .
```

The image contains the exact LightlyTrain source state in the local checkout at
build time; it does not download LightlyTrain from GitHub.

## Run

Mount a directory containing models or exported ONNX files at `/workspace`:

```bash
docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --volume "$PWD":/workspace \
  lightly-train-migraphx
```

`--security-opt seccomp=unconfined` is optional, but AMD recommends it for
ROCm containers in HPC environments.

## Verify the installation

Inside the container, verify that both Python APIs are available:

```bash
python3 -c "import lightly_train, migraphx; print('LightlyTrain and MIGraphX are available')"
```

On a host with a supported AMD GPU, this additionally checks that MIGraphX can
create its GPU compilation target:

```bash
python3 -c "import migraphx; print(migraphx.get_target('gpu'))"
```

## Scope

This is intentionally a minimal export/inference environment. PyTorch is
installed as a dependency of LightlyTrain and is not guaranteed to be a
ROCm-enabled build. Use an AMD `rocm/pytorch` base image instead if you need to
train or run PyTorch workloads on the GPU inside this container.
