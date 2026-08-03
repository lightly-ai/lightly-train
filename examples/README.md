# Examples

This directory contains runnable examples for LightlyTrain. Examples are grouped
by how they are intended to be used:

- [`notebooks/`](notebooks/) contains Python notebooks designed for local or
  hosted notebook environments. These install their dependencies with pip and
  are integrated into the documentation site.
- [`deployment/`](#future-deployment-layout) is the home for new non-notebook
  export, inference, and serving recipes. It is described below and will be
  adopted incrementally to later match the layout given further down.

## Future deployment layout

New deployment recipes use a language-first layout. This keeps reusable code
in the same language close to its consumers, while still making the target
platform and runtime explicit:

```text
deployment/
  cpp/
    common/
    cuda-onnxruntime/
    cuda-tensorrt/
    rocm-migraphx/
  python/
    common/
    cuda-onnxruntime/
    cuda-tensorrt/
    rocm-migraphx/
  javascript/
    openvino/
  swift/
    apple/
      coreml/
  servers/
    triton/
      model_repository/
      clients/
        python/
        cpp/
        javascript/
```

The platform level is used when it meaningfully affects the toolchain or
artifact (for example `cuda`, `rocm`, `android`, or `apple`). A runtime that
does not need one may sit directly below its language, as in
`c/openvino/`. Add a target-specific level later if it becomes necessary;
do not create an artificial level just for uniformity.

For example, a new C++ MIGraphX recipe belongs in
`deployment/cpp/rocm/migraphx/`. A LiteRT recipe belongs below the language
and target it actually uses, rather than in one shared LiteRT source tree.

### Shared code

Each runtime adapter is self-contained: its build configuration, dependencies,
and runtime glue live with the recipe. Share code only when it is both
model-neutral and runtime-neutral, and only within the same language. For
example, C++ image processing or detection postprocessing may live in
`deployment/cpp/common/`; CUDA, ROCm, TensorRT, MIGraphX, and LiteRT adapters
must not be dependencies of that common code. Do not add cross-language shared
implementation layers.

### Servers

Inference servers are organized by serving system, not by client language or
backend. Triton therefore belongs in `deployment/servers/triton/`, with its
model repository and server configuration together. Client examples live in
`clients/<language>/`. A Triton recipe may demonstrate multiple backends, but
backend-specific direct inference remains in that backend's ordinary runtime
recipe.

### Environments

Development environments remain centralized in [`.devcontainer/`](../.devcontainer/)
and are named for the toolchain they install, rather than for an individual
language or test suite. One CUDA environment can support both ONNX Runtime's
CUDA execution provider and TensorRT; one ROCm/MIGraphX environment can
support both Python and C++ examples. Add dedicated environments for Triton,
OpenVINO, LiteRT/Android, or CoreML only when their toolchains cannot
reasonably share an existing one.

Each deployment recipe must state its required dev container or Docker image
in its README. This index must link each environment back to the recipes it
supports, so examples remain discoverable from either their language or their
runtime/server.

## Runtime and server index

| Runtime or server | Future recipe locations |
| --- | --- |
| ONNX Runtime CUDA EP | `deployment/cpp/cuda/onnxruntime/` |
| TensorRT | `deployment/cpp/cuda/tensorrt/` |
| MIGraphX | `deployment/cpp/rocm/migraphx/`, `deployment/python/rocm/migraphx/` |
| LiteRT | `deployment/cpp/android/litert/`, `deployment/python/host/litert/`, `deployment/c/embedded/litert/` |
| OpenVINO | `deployment/c/openvino/`, `deployment/javascript/openvino/` |
| Core ML | `deployment/swift/apple/coreml/` |
| Triton Inference Server | `deployment/servers/triton/` |
