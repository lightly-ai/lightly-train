# Inference Benchmarks

Benchmarks LightlyTrain's COCO-pretrained object detection checkpoints for inference
speed and accuracy across different export backends and precisions.

## TensorRT Benchmark

Runs every COCO-pretrained detection checkpoint through TensorRT export (fp32 and fp16)
and evaluates it on the COCO val2017 split, reporting mAP, latency, and throughput.

Benchmarked models (see `MODEL_NAMES` in
[run_tensorrt_benchmark.py](./run_tensorrt_benchmark.py)):

- `picodet-s-coco`, `picodet-l-coco`
- `ltdetrv2-s-coco`
- `dinov3/vitt16-ltdetr-coco`, `dinov3/vitt16plus-ltdetr-coco`,
  `dinov3/vits16-ltdetr-coco`
- `dinov3/convnext-{tiny,small,base,large}-ltdetr-coco`
- `dinov2/vits14-noreg-ltdetr-coco`

### Requirements

- A CUDA GPU with TensorRT support.
- A local copy of the COCO 2017 dataset (`annotations/instances_{train,val}2017.json`
  and the `train2017/`/`val2017/` image directories).
- [uv](https://docs.astral.sh/uv/) for dependency management.

### Running

```bash
make benchmark-tensorrt COCO_ROOT=/path/to/coco
```

This runs `run_tensorrt_benchmark.py` with the pinned dependencies from
`requirements/tensorrt.txt`. `COCO_ROOT` defaults to `/datasets/coco` (see
[Makefile](./Makefile)).

To run the script directly with more control over its options:

```bash
uv run --frozen --with-requirements requirements/tensorrt.txt run_tensorrt_benchmark.py \
    --coco-root /path/to/coco \
    --out out/tensorrt_benchmark \
    --batch-size 1 \
    --warmup-steps 5 \
    --steps 10  # limit batches per run, useful for smoke-testing
```

### Output

Results are written to `out/tensorrt_benchmark/<model>/<precision>/`:

- `model.onnx` / `model.engine` — the exported ONNX and TensorRT artifacts.
- `benchmark_results.json` — raw metrics and timing data.
- `benchmark_summary.md` — human-readable report for that model/precision.

A combined `out/tensorrt_benchmark/summary.md` aggregates mAP, latency, and throughput
for every model/precision combination into a single table.

## Torch SAHI Benchmark

Runs the three `ltdetrv2` checkpoints (`ltdetrv2-{s,m,l}-coco`) through the torch
backend with `torch.compile` and bf16-mixed autocast on CUDA, under several amounts of
SAHI tiling. The settings are named `sahi-<tile_size>-<overlap>`:

- `no-sahi` — no tiling.
- `sahi-640-0.2`, `sahi-640-0.6` — tiles cut at the model's 640x640 input, so they are
  native-resolution crops and nothing is magnified. This isolates the effect of tiling
  alone.
- `sahi-320-0.2`, `sahi-320-0.6` — tiles cut at 320x320 and resized up to the model's
  input, so every tile is magnified 2x. This is the `predict_sahi()` default and where
  the gain on small objects comes from, at roughly three times the tiles of the 640
  setting at the same overlap.

`tile_size` is the region of the original image a tile covers, and the tile is resized to
the model's input before the forward pass — so a smaller `tile_size` means more
magnification *and* more tiles. `overlap` controls how densely the tiles are laid down.
Both drive latency, which is why the sweep varies them separately.

All runs use a score threshold of `0.05`, so the mAP values are comparable across
settings.

### Requirements

- A CUDA GPU.
- A detection dataset in yolo or coco format, provided as a data YAML file. **The images
  should be considerably larger than a tile**: an image that fits inside a single tile is
  not tiled at all and collapses onto the untiled baseline, so on a dataset with small
  images the curves become indistinguishable. VisDrone2019-DET is a good fit.
- The class ids in the data config must line up with the class ids of the
  COCO-pretrained checkpoints. The benchmark matches ground truth against predictions by
  class id and does not remap them by name, so a mismatched class list silently yields a
  meaningless mAP. For VisDrone, use `remap_visdrone_to_coco.py`, see
  [COCO class id remapping](#coco-class-id-remapping).
- [uv](https://docs.astral.sh/uv/) for dependency management.

### COCO class id remapping

The original VisDrone2019-DET dataset has its own label space (`0 pedestrian`, ...,
`9 motor`), which means something entirely different to a COCO-pretrained checkpoint:
VisDrone's `4 van`, for example, is scored against COCO's `4 airplane`. On the same 32
images, `ltdetrv2-l-coco` reports mAP `0.018` on the original data config and `0.211` on
the remapped one.

`remap_visdrone_to_coco.py` writes a remapped copy of the dataset. The `images`
directories are symlinked to the originals, the `labels` files are rewritten with COCO
class ids, and a `*_coco_remapped.yaml` data config is generated:

```bash
make remap-visdrone VISDRONE_ROOT=/path/to/visdrone \
    VISDRONE_COCO_ROOT=/path/to/visdrone_coco_remapped
```

VisDrone classes with no COCO counterpart (`tricycle`, `awning-tricycle`) are dropped,
`pedestrian` and `people` both become `person`, and `van` becomes `car`. The mapping is
a single constant at the top of the script.

The generated config lists all 80 COCO classes even though only six of them occur in the
labels. This is load-bearing: the yolo dataset maps the keys of `names` to internal
class ids by enumerating them in insertion order, so a sparse subset such as
`{0, 1, 2, 3, 5, 7}` would be compacted to `0..5` and stop lining up with the ids the
checkpoints predict. Classes without ground truth are excluded from the mAP, so the 74
unused names cost nothing.

### Running

```bash
make benchmark-torch-sahi VISDRONE_DATA=/path/to/visdrone_coco_remapped/visdrone_coco_remapped.yaml
```

This runs `run_torch_sahi_benchmark.py` with the pinned dependencies from
`requirements/tensorrt.txt`. `VISDRONE_DATA` defaults to
`/datasets/visdrone_coco_remapped/visdrone_coco_remapped.yaml` (see
[Makefile](./Makefile)). This benchmark does not use TensorRT, but it shares that
environment rather than pinning a second one: it is a superset of what the torch backend
needs.

To run the script directly with more control over its options:

```bash
uv run --frozen --with-requirements requirements/tensorrt.txt run_torch_sahi_benchmark.py \
    --data /path/to/visdrone_coco_remapped/visdrone_coco_remapped.yaml \
    --dataset-name "VisDrone2019-DET val" \
    --out out/torch_sahi_benchmark \
    --batch-size 1 \
    --warmup-steps 20 \
    --steps 10  # limit batches per run, useful for smoke-testing
```

Warmup defaults to 20 batches rather than 5: with SAHI the number of tiles differs
between images, so `torch.compile` re-traces the graph until it settles on a dynamic
batch dimension, and those recompilations must land outside the measured window.

### Output

Results are written to `out/torch_sahi_benchmark/<model>/<sahi-setting>/`:

- `benchmark_results.json` — raw metrics and timing data.
- `benchmark_summary.md` — human-readable report for that model/SAHI setting.

Aggregated over all runs, at `out/torch_sahi_benchmark/`:

- `summary.md` — mAP (overall, small, and large), latency, and throughput for every
  model/SAHI combination.
- `latency_vs_map.png` — latency vs. mAP@0.5:0.95, with one curve per SAHI setting
  running through the small, medium, and large model. The latency axis is logarithmic:
  the settings span more than an order of magnitude, so a linear axis squashes every
  cheap setting into the left edge.
- `latency_vs_map_small.png` — the same plot for small objects only. Tiling exists for
  those, and the overall mAP averages the effect away against the medium and large ones,
  so this is the plot that shows what a setting actually bought.

### Running on ROCm

The torch SAHI benchmark also runs on AMD GPUs, using a second environment pinned in
`requirements/rocm.txt`:

```bash
make benchmark-torch-sahi-rocm
```

Results go to `out/torch_sahi_benchmark_rocm/` so that they do not overwrite the CUDA
ones. Note that the reports and the plot title still say CUDA: ROCm builds of torch
expose the GPU as the `cuda` device, and the benchmark takes the label from there.

Differences to `requirements/tensorrt.txt`:

- `lightly-train` is installed **editable from the local checkout** rather than from the
  git URL, so the benchmark measures the working tree.
- No TensorRT, since there is no ROCm build of it. Only the torch backend is available.
- `torch` and `torchvision` are pinned to the `+rocm7.2` builds from the PyTorch index,
  matching the `pinned-rocm-torch` dependency group in the repository's
  `pyproject.toml`. These wheels are newer than the `--exclude-newer` date used for the
  TensorRT lock, so the ROCm lock has its own `EXCLUDE_NEWER_ROCM` date.

## Managing dependencies

Never edit the `.txt` files in `requirements/` directly. Instead, edit the corresponding
`.in` file and run `make lock` to regenerate the pinned `.txt` files (or
`make lock-tensorrt` / `make lock-rocm` for one of them).
