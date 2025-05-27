(performance)=

# Performance

Lightly**Train** was built to have good performance out of the box. It is built upon
[PyTorch](https://github.com/pytorch/pytorch) and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and thus benefits from the performance optimizations
of these libraries. A performance example, when using Lightly**Train** on two NVIDIA RTX 4090 is provided in {ref}`hardware-recommendations`.

However, there are still ways to improve the performance of Lightly**Train** by adjusting
it to specific hardware and use cases. The following recommendations apply to both the
{meth}`lightly_train.train` and {meth}`lightly_train.embed` commands.

(speeding-up-the-model-part)=

## Speeding Up the Model Part

Speeding up the model part of the training is usually the most effective way to improve performance.

To find out if the model part is indeed the bottleneck, see {ref}`finding-the-performance-bottleneck`.

(accelerators)=

### Using Accelerators (GPUs, TPUs, Etc.)

Lightly**Train** supports training on CPUs and GPUs. Support for other accelerators is experimental.
By default, Lightly**Train** will use the best available accelerator.
To use a different accelerator, set, for example, the `accelerator=cpu` argument.

### Multi-GPU

See {ref}`multi-gpu` for information on how to train on multiple GPUs.
By default, all available GPUs are used for training.

### Multi-Node

See {ref}`multi-node` for information on how to train on multiple nodes.
By default, a single node is used for training.

### Mixed Precision

Set `precision="16-mixed"` to enable mixed/half precision training. See {meth}`lightly_train.train`
for all available precision options.

## Using Newest Dependencies

Lightly**Train** can be made faster by using newer versions of its dependencies, which might contain performance improvements.

**Recommendations:**

- When using GPUs, install the latest versions of the NVIDIA drivers, CUDA, and cuDNN.
- Install the latest versions of [PyTorch](https://github.com/pytorch/pytorch),
  [TorchVision](https://github.com/pytorch/vision), and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). Make sure that they were built with support for your CUDA version.
- Install newer versions of Python.

(finding-the-performance-bottleneck)=

## Finding the Performance Bottleneck

While training, Lightly**Train** shows a `data_wait` percentage in the progress bar
alongside the training loss. `data_wait` is the percentage of time spent
waiting for new data to be loaded before passing it to the model. The value should be
close to zero; a high value indicates that data loading is the bottleneck in the training
process.

The `data_wait` percentage is calculated as out of the `batch_time` and the `data_time` as\
`data_time / (batch_time + data_time)`.

The `batch_time` is the time in seconds taken by the main process for the forward, backward, and optimizer step.
It uses the accelerator(s) like GPUs if available.

The `data_time` is the time in seconds the main process waits while fetching the next batch
from the dataloading workers. As the dataloading workers run in parallel and already
prepare the next batch while the current batch is processed, the `data_time` should
be close to zero.

Both the `batch_time` and the `data_time` are visible in the MLflow, TensorBoard, and Weights & Biases logs.

### Model Bottleneck

A `data_wait` ratio \<10% means that almost all of the time is spent in the
model forward and backward pass and data loading is not the bottleneck.
This usually shows in the accelerator utilization being high, e.g., shown
by `nvidia-smi` for GPUs.
To speed up this step, see the section {ref}`speeding-up-the-model-part`.

### Dataloader Bottleneck

If the `data_wait` ratio is >10%, the dataloading should be optimized.
To find out if loading the images from disk (I/O-bound) or the
decoding and augmentations (CPU-bound) are the bottleneck, run `top` or `htop`.

In any case, data loading can be sped up by converting the images into a format
with faster decoding, such as JPEG or WebP. Additionally, lowering their resolution
can help. Both approaches are especially useful if the images are stored in a format
that is slow to decode, such as PNG.

#### Dataloader Bottleneck: CPU-Bound

If the CPU is fully utilized, the bottleneck is due to image decoding and augmentations.

Reducing the `num_workers` parameter might help, as it increases cache locality and reduces the overhead of process switches.

#### Dataloader Bottleneck: I/O-Bound

If the CPU is not fully utilized, the bottleneck is the I/O. This can be improved by
moving the images to local fast storage, e.g., an SSD, or by using a faster network.

Furthermore, the `num_workers` parameter should be
increased to allow more dataloading workers to run in parallel.
By default, the `num_workers` parameter is set to `min((num_CPU_cores - num_devices) / num_devices, 8)`.
This ensures that `num_workers` * `num_devices` is as large as possible while
not overloading the CPU with more total workers than cores. The default number of workers
is capped at 8 workers per device to avoid excessive CPU usage on systems with many
CPU cores. The maximum value can be configured with the `LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO`
environment variable. No maximum is applied if `num_workers` is set manually.

```{toctree}
---
hidden:
maxdepth: 1
---
multi_gpu
multi_node
hardware_recommendations
```
