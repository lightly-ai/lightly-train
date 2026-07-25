# LightlyTrain with MIGraphX

This example provides a container with AMD ROCm, MIGraphX, and the ONNX export
dependencies of LightlyTrain. It avoids installing ROCm or MIGraphX in the host
userspace, and installs LightlyTrain from the local repository source used as the Docker
build context.

The host still needs a supported AMD GPU driver and access to `/dev/kfd` and `/dev/dri`.
MIGraphX is installed from AMD's ROCm apt repository because it is not available as a
standalone PyPI package.

## Build

From the repository root:

```bash
docker build --tag lightly-train-migraphx -f examples/migraphx/Dockerfile .
```

The image contains the exact LightlyTrain source state in the local checkout at build
time; it does not download LightlyTrain from GitHub.

## Run

All commands below run from the repository root and mount it at `/workspace`, so the
scripts in `examples/migraphx/` and any exported artifacts are shared between the host
and the container:

```bash
docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --volume "$PWD":/workspace \
  lightly-train-migraphx \
  <command>
```

`--security-opt seccomp=unconfined` is optional, but AMD recommends it for ROCm
containers in HPC environments. Omit `<command>` (and drop `-it` for `--rm -i`, or keep
`-it`) to get an interactive shell instead.

## Verify the installation

Inside the container, verify that the required Python APIs are available:

```bash
python -c "import lightly_train, migraphx, onnx, onnxruntime; print('LightlyTrain and MIGraphX are available')"
```

On a host with a supported AMD GPU, this additionally checks that MIGraphX can create
its GPU compilation target:

```bash
python -c "import migraphx; print(migraphx.get_target('gpu'))"
```

## Export an LT-DETR model to MIGraphX

`export_model.py` downloads an LT-DETR checkpoint, exports a fully static
batch-size-1 ONNX model, and compiles it for the available AMD GPU:

```bash
docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --volume "$PWD":/workspace \
  lightly-train-migraphx \
  python /workspace/examples/migraphx/export_model.py \
    --out /workspace/dinov3-vitt16-ltdetr-coco.mxr
```

This exports the 38.7 MB `dinov3/vitt16-ltdetr-coco` checkpoint by default (override
with `--checkpoint`). It creates both `/workspace/dinov3-vitt16-ltdetr-coco.onnx` and
`/workspace/dinov3-vitt16-ltdetr-coco.mxr`. Compilation is hardware-dependent and can
take a minute or longer.

## Download an example image

```bash
wget -O image.jpg http://images.cocodataset.org/val2017/000000577932.jpg
```

## Run inference and plot the detections

`predict.py` loads the compiled engine, reads its expected input size directly from
the engine, runs inference on an image (with hardcoded ImageNet normalization), and
saves a plot of the predicted bounding boxes:

```bash
docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --volume "$PWD":/workspace \
  lightly-train-migraphx \
  python /workspace/examples/migraphx/predict.py \
    --engine /workspace/dinov3-vitt16-ltdetr-coco.mxr \
    --image /workspace/image.jpg \
    --out /workspace/prediction.jpg
```

`prediction.jpg` is written to the mounted `$PWD` on the host. `predict.py` also loads
`--checkpoint` (the same alias used for export, by default) on the CPU to look up class
names for the predicted labels; the engine itself only returns raw class logits and
normalized boxes.

## Scope

This is intentionally a minimal export/inference environment. PyTorch is installed as a
dependency of LightlyTrain and is not guaranteed to be a ROCm-enabled build. Both
scripts load the PyTorch model on CPU (for export, and for label lookup during
prediction) and use the GPU only for MIGraphX compilation and inference. Use an AMD
`rocm/pytorch` base image instead if you need to train or run PyTorch workloads on the
GPU.
