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

## Managing dependencies

Never edit the `.txt` files in `requirements/` directly. Instead, edit the corresponding
`.in` file and run `make lock` to regenerate the pinned `.txt` file.
