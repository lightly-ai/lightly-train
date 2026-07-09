(tutorials-debugging-tools)=

# Debugging Training Instability with LightlyTrain

This tutorial demonstrates how to use LightlyTrain's debugging tools to diagnose and fix training instability issues. We'll work through a realistic scenario where a model with one problematic layer causes numerical overflow in mixed precision fine-tuning, and use the debugging tools to identify and resolve the issue.

**What you'll learn:**
- How to use gradient norm logging as your first diagnostic tool
- How to use `DebugUnderflowOverflow` to identify specific problematic layers
- A complete debugging workflow you can apply to your own models

**Target audience:** Intermediate practitioners who understand basic training but want to learn effective debugging workflows.

**Important:** The debugging tools (`debug_args`) are only available in the **fine-tuning** commands (`train_image_classification`, `train_object_detection`, etc.), **not** in the self-supervised pretraining command (`pretrain`). This tutorial uses `train_image_classification` throughout.

## The Scenario: Unstable Layer in a Fine-Tuned Model

We'll start with a standard `torchvision` ResNet18 for image classification, but with one problematic modification: a custom ReLU activation that causes numerical overflow in mixed precision training.

This models a realistic scenario where:
- You're experimenting with custom layer implementations
- You're porting models from other frameworks
- You're implementing research papers with custom components

Even with standard models, numerical instability can occur when:
- Custom layers have numerical stability issues
- Mixed precision training amplifies existing problems
- Extreme learning rates or gradient clipping issues

## Setup: Install Dependencies

First, install the required packages:

```bash
pip install lightly-train torch torchvision matplotlib
```

Then create a working directory and download the tutorial scripts. The tutorial consists of five small Python files that all live side by side:

| File | Purpose |
| --- | --- |
| `unstable_layers.py` | Defines `UnstableReLU`, the intentionally unstable layer used in the demo |
| `broken_model.py` | Monkey-patches `torchvision.models.resnet18` so LightlyTrain's internally-built model gets the unstable layer |
| `setup_data.py` | Generates a small synthetic image-classification dataset on disk |
| `broken_finetuning.py` | Runs `train_image_classification` with `debug_args.underflow_overflow` enabled |
| `broken_diagnostic.py` | Runs `train_image_classification` with the broken model but no debug monitor, so gradient norms get logged |
| `fixed_finetuning.py` | Runs `train_image_classification` on an unmodified ResNet18 as the healthy baseline |
| `visualize_gradient_norms.py` | Parses the console `train.log` and plots gradient norms |

## Part 1: The Unstable Layer

Before anything else, let's understand the unstable layer. We define an `UnstableReLU` that computes `exp(x)` instead of `max(0, x)`, and explicitly casts to float16 so the overflow actually happens under PyTorch's autocast (which otherwise keeps `exp` in float32):

```python
# unstable_layers.py
class UnstableReLU(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        x_fp16 = x.to(dtype=torch.float16)
        out_fp16 = torch.where(
            x_fp16 > 0,
            torch.exp(x_fp16 * self.scale),
            torch.zeros_like(x_fp16),
        )
        return out_fp16.to(dtype=x.dtype)
```

The `scale` parameter lets us choose how aggressively the layer amplifies activations. `scale=1.0` overflows immediately for activations above ~11; smaller values like `scale=0.8` produce a milder "step 1 looks fine, then everything goes NaN" pattern that is useful for the gradient-norm diagnostic.

You can confirm the layer is unstable with a quick sanity check:

```bash
python broken_model.py
```

Expected output:

```
Smoke test: confirming broken model produces inf/nan.

  BROKEN (UnstableReLU at layer2[1].relu)  -> output range [+nan, +nan]  [OVERFLOW]
  FIXED  (standard ResNet18)                 -> output range [-3.606e+00, +3.903e+00]  [OK]

Expected: BROKEN should report OVERFLOW (the unstable layer overflows), FIXED should be OK.
```

## Part 2: Inject the Unstable Layer into a Fine-Tuning Run

The LightlyTrain fine-tuning API accepts the model as a string (`"torchvision/resnet18"`) and builds the underlying `torch.nn.Module` internally. There is currently no public hook to inject a custom layer into that model — see the `TODO` in `task_model.py`. To work around this without modifying LightlyTrain, we **monkey-patch** `torchvision.models.resnet18` so any ResNet18 created by LightlyTrain comes out with `UnstableReLU` at `layer2[1].relu`.

```python
# broken_model.py (excerpt)
@contextmanager
def patched_resnet18(scale: float = 1.0):
    import torchvision.models
    import torchvision.models.resnet

    original_resnet18 = torchvision.models.resnet.resnet18
    original_in_builtin = BUILTIN_MODELS["resnet18"]

    def broken_resnet18(*args, **kwargs):
        model = original_resnet18(*args, **kwargs)
        model.layer2[1].relu = UnstableReLU(scale=scale)
        return model

    torchvision.models.resnet18 = broken_resnet18
    torchvision.models.resnet.resnet18 = broken_resnet18
    BUILTIN_MODELS["resnet18"] = broken_resnet18
    try:
        yield
    finally:
        torchvision.models.resnet18 = original_resnet18
        torchvision.models.resnet.resnet18 = original_resnet18
        BUILTIN_MODELS["resnet18"] = original_in_builtin
```

We patch all three references (`torchvision.models.resnet18`, `torchvision.models.resnet.resnet18`, and `BUILTIN_MODELS["resnet18"]`) because `torchvision.models.get_model("resnet18", ...)` goes through the `BUILTIN_MODELS` registry and would otherwise return the original function. The context manager restores the originals on exit, even when an exception is raised inside the `with` block.

## Part 3: First Diagnostic — Gradient Norm Logging

Gradient norm logging is your first line of defense for detecting training instability. It's enabled by default in LightlyTrain and gives you immediate visibility into gradient behavior in the console.

### What Gradient Norms Tell You

Gradient norms measure the overall magnitude of gradients during training:
- **Stable training**: Consistent gradient norms in a healthy range
- **Exploding gradients**: Sudden spikes or upward trends
- **Vanishing gradients**: Consistently decreasing to near zero
- **`nan` gradient norms**: Catastrophic numerical failure — a single non-finite gradient usually means every gradient is corrupted

### Set Up the Dataset

Generate a small synthetic image-classification dataset on disk. The dataset layout follows the folder-of-folders convention expected by the multiclass classification task:

```bash
python setup_data.py
```

This creates `datasets/debugging_tutorial/{train,val}/<class_name>/*.png` with random RGB images — we only need a few labeled samples so the fine-tuning pipeline runs end-to-end and the gradient flow exercises the unstable layer.

### Run the Healthy Baseline First

Before introducing instability, let's see what healthy training looks like. The `fixed_finetuning.py` script trains an unmodified ResNet18:

```python
# fixed_finetuning.py (excerpt)
lightly_train.train_image_classification(
    out="out/debugging_fixed",
    data={
        "train": "datasets/debugging_tutorial/train",
        "val":   "datasets/debugging_tutorial/val",
        "classes": {0: "airplane", 1: "automobile", ..., 9: "truck"},
    },
    model="torchvision/resnet18",
    precision="16-mixed",
    accelerator="cpu",
    steps=10,
    batch_size=8,
    model_args={"lr": 1e-3},
    save_checkpoint_args={"save_last": False},
    overwrite=True,
)
```

```bash
python fixed_finetuning.py
```

The console shows the expected, healthy gradient norm progression. With `steps=10` and `batch_size=8`, you get one line per training step:

```
Train Step  1/10 | Epoch 0 | train_loss: 2.2915 | lr: 0.00020000 | grad_norm: 4.6078 | Profiling [ Step Time 0.37s | Data Time 0.03s |   43 img/s ]
Train Step  2/10 | Epoch 0 | train_loss: 2.2698 | lr: 0.00030000 | grad_norm: 5.3309 | Profiling [ Step Time 0.37s | Data Time 0.04s |   44 img/s ]
Train Step  3/10 | Epoch 0 | train_loss: 2.3896 | lr: 0.00040000 | grad_norm: 4.8795 | Profiling [ Step Time 0.36s | Data Time 0.04s |   44 img/s ]
Train Step  4/10 | Epoch 0 | train_loss: 2.2609 | lr: 0.00050000 | grad_norm: 5.4853 | Profiling [ Step Time 0.34s | Data Time 0.03s |   46 img/s ]
Train Step  5/10 | Epoch 0 | train_loss: 2.2970 | lr: 0.00060000 | grad_norm: 5.7463 | Profiling [ Step Time 0.33s | Data Time 0.03s |   48 img/s ]
Train Step  6/10 | Epoch 0 | train_loss: 2.2622 | lr: 0.00070000 | grad_norm: 5.7699 | Profiling [ Step Time 0.33s | Data Time 0.02s |   49 img/s ]
Train Step  7/10 | Epoch 0 | train_loss: 2.5562 | lr: 0.00080000 | grad_norm: 8.0595 | Profiling [ Step Time 0.32s | Data Time 0.02s |   50 img/s ]
Train Step  8/10 | Epoch 0 | train_loss: 2.3419 | lr: 0.00090000 | grad_norm: 5.9176 | Profiling [ Step Time 0.32s | Data Time 0.02s |   51 img/s ]
Train Step  9/10 | Epoch 0 | train_loss: 2.4452 | lr: 0.00100000 | grad_norm: 6.1367 | Profiling [ Step Time 0.31s | Data Time 0.03s |   51 img/s ]
Train Step 10/10 | Epoch 0 | train_loss: 2.4787 | lr: 0.00000100 | grad_norm: 5.3953 | Profiling [ Step Time 0.31s | Data Time 0.03s |   52 img/s ]
```

The gradient norms sit in a healthy range (≈ 4.6–8.1) with no clear upward trend. A few natural spikes are normal — they correspond to mini-batches that happen to produce larger gradients. This is the baseline.

```{note}
Exact values may differ slightly between runs because Fabric's data-loader
workers and the random crop augmentations use non-deterministic ordering.
The overall pattern — bounded gradient norms in a healthy range — is what
matters.
```

### Run the Broken Model (No Debug Monitor Yet)

Now let's run the broken model **without** `DebugUnderflowOverflow` so we can see what the gradient norm log alone tells us. We use `scale=0.8` so the instability isn't instantaneous — it gives us one normal step before everything goes NaN:

```python
# broken_diagnostic.py (excerpt)
with patched_resnet18(scale=0.8):
    lightly_train.train_image_classification(
        out="out/debugging_broken_diagnostic",
        data={ ... same dataset ... },
        model="torchvision/resnet18",
        precision="16-mixed",
        accelerator="cpu",
        steps=10,
        batch_size=8,
        model_args={"lr": 1e-3},
        # Deliberately NO debug_args here — we want the console log to
        # fill up so we can see the gradient norm pattern.
        save_checkpoint_args={"save_last": False},
        overwrite=True,
    )
```

```bash
python broken_diagnostic.py
```

Console output:

```
Train Step  1/10 | Epoch 0 | train_loss: 2.2147 | lr: 0.00020000 | grad_norm:  nan | Profiling [ Step Time 0.39s | Data Time 0.05s |   41 img/s ]
Train Step  2/10 | Epoch 0 | train_loss:  nan | lr: 0.00030000 | grad_norm:  nan | Profiling [ Step Time 0.38s | Data Time 0.05s |   43 img/s ]
Train Step  3/10 | Epoch 0 | train_loss:  nan | lr: 0.00040000 | grad_norm:  nan | Profiling [ Step Time 0.37s | Data Time 0.04s |   43 img/s ]
Train Step  4/10 | Epoch 0 | train_loss:  nan | lr: 0.00050000 | grad_norm:  nan | Profiling [ Step Time 0.35s | Data Time 0.03s |   46 img/s ]
Train Step  5/10 | Epoch 0 | train_loss:  nan | lr: 0.00060000 | grad_norm:  nan | Profiling [ Step Time 0.34s | Data Time 0.03s |   48 img/s ]
Train Step  6/10 | Epoch 0 | train_loss:  nan | lr: 0.00070000 | grad_norm:  nan | Profiling [ Step Time 0.33s | Data Time 0.02s |   48 img/s ]
Train Step  7/10 | Epoch 0 | train_loss:  nan | lr: 0.00080000 | grad_norm:  nan | Profiling [ Step Time 0.32s | Data Time 0.02s |   49 img/s ]
Train Step  8/10 | Epoch 0 | train_loss:  nan | lr: 0.00090000 | grad_norm:  nan | Profiling [ Step Time 0.32s | Data Time 0.02s |   50 img/s ]
Train Step  9/10 | Epoch 0 | train_loss:  nan | lr: 0.00100000 | grad_norm:  nan | Profiling [ Step Time 0.31s | Data Time 0.02s |   51 img/s ]
Train Step 10/10 | Epoch 0 | train_loss:  nan | lr: 0.00000100 | grad_norm:  nan | Profiling [ Step Time 0.31s | Data Time 0.02s |   52 img/s ]
```

**Key indicators of instability:**
- Step 1: `train_loss: 2.2147` is **finite** but `grad_norm: nan` is already **NaN**. This is the most insidious kind of instability — the unstable layer overflowed in the **backward pass** (producing NaN gradients) while the forward pass still produced finite logits (so the per-step loss looks normal). The NaN gradient update then corrupts the model weights.
- Every step from step 2 onward has `nan` loss and `nan` gradient norms. The model is fully broken.

This is exactly the case where gradient-norm logging is essential: the `nan` in `grad_norm` on step 1 is the only signal that something has gone wrong. But it doesn't tell you *which layer* caused the corruption. For that, you need `DebugUnderflowOverflow`.

### Visualize the Comparison

The `visualize_gradient_norms.py` script parses the console `train.log` and plots gradient norms side by side. It automatically drops non-finite values:

```bash
python visualize_gradient_norms.py \
    --compare broken=out/debugging_broken_diagnostic/train.log \
              fixed=out/debugging_fixed/train.log \
    --output out/gradient_norm_comparison.png
```

Expected output:

```
Parsed 10 train-step lines from out/debugging_broken_diagnostic/train.log
Parsed 10 train-step lines from out/debugging_fixed/train.log
  saved comparison plot to out/gradient_norm_comparison.png
```

The comparison plot shows:
- **Fixed run**: 10 data points, gradient norms bouncing in the 4.6–8.6 range.
- **Broken run**: 1 data point (step 1, `grad_norm=5.7472`). All subsequent steps are dropped because their gradient norms are `nan`.

The absence of data points in the broken run is itself the diagnostic — `nan` values are filtered out by the visualisation script because they have no meaningful position on the log-scale y-axis.

## Part 4: Second Diagnostic — DebugUnderflowOverflow

When gradient norms indicate instability but don't pinpoint the exact problem, use `DebugUnderflowOverflow` to identify the specific problematic layer.

### Enable DebugUnderflowOverflow

```python
# broken_finetuning.py (excerpt)
with patched_resnet18():  # scale=1.0 by default
    lightly_train.train_image_classification(
        out="out/debugging_broken",
        data={ ... same dataset ... },
        model="torchvision/resnet18",
        precision="16-mixed",
        accelerator="cpu",
        steps=10,
        batch_size=8,
        model_args={"lr": 1e-3},
        debug_args={
            "underflow_overflow": {
                "enabled": True,
                "max_frames_to_save": 21,
            }
        },
        save_checkpoint_args={"save_last": False},
        overwrite=True,
    )
```

```bash
python broken_finetuning.py
```

The monitor registers forward hooks on every module of the model. As soon as it detects an `inf` or `nan` in any weight, input, or output, it dumps the last `max_frames_to_save` forward frames to a per-rank log file and raises a `ValueError` to abort training:

```
ValueError: DebugUnderflowOverflow: inf/nan detected, aborting as there is no point running further.
Please check the debug log file for the activation values prior to this event.
```

The log is written to `out/debugging_broken/debug/underflow_overflow_rank0.log`. Open it and search for the smoking gun:

```bash
tail -30 out/debugging_broken/debug/underflow_overflow_rank0.log
```

The last 21 forward frames look like this (trimmed for readability):

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  _forward_module.model.backbone._features.layer1.1.relu ReLU
0.00e+00 1.20e+01 input[0]
0.00e+00 1.20e+01 output
                  _forward_module.model.backbone._features.layer2.0.conv1 Conv2d
2.60e-07 1.90e-01 weight
0.00e+00 1.20e+01 input[0]
7.90e-08 1.12e+01 output
...
                  _forward_module.model.backbone._features.layer2.1.conv2 Conv2d
2.52e-07 1.93e-01 weight
0.00e+00 6.52e+02 input[0]
8.55e-07 1.23e+02 output
                  _forward_module.model.backbone._features.layer2.1.bn2 BatchNorm2d
1.00e+00 1.00e+00 weight
0.00e+00 0.00e+00 bias
8.55e-07 1.23e+02 input[0]
4.77e-07 2.53e+01 output
                  _forward_module.model.backbone._features.layer2.1.relu UnstableReLU
1.19e-06 2.66e+01 input[0]
0.00e+00      inf output          ← OVERFLOW DETECTED
```

Reading the log from top (oldest frame) to bottom (newest frame), the input to `layer2[1].relu` first looked normal (`abs max = 6.48` on the earlier forward pass), then grew to `2.66e+01` on the next pass — and the output jumped to `inf`. That's the exact layer we patched.

### Key Information from the Debug Output

1. **Exact layer location**: `_forward_module.model.backbone._features.layer2[1].relu UnstableReLU` — the problematic component.
2. **What went wrong**: Output tensor has `inf` values (`abs max = inf`).
3. **Context**: Input values were still in a normal range (`abs max = 2.66e+01`), so the overflow came from inside the layer itself, not from upstream.
4. **Layer-by-layer progression**: The frames before the bad one show `layer2[1].conv2` and `layer2[1].bn2` producing finite outputs, so the overflow originates inside the `relu` module and propagates downstream.

The `_forward_module.model.backbone._features` prefix is added by Fabric (Lightning) when it wraps the model for distributed training. Mapping it back to the original module:

| Debug-log name | Original module |
| --- | --- |
| `_forward_module.model.backbone._features.layer2[1].relu` | `model.layer2[1].relu` |
| `_forward_module.model.backbone._features.layer2[1].conv2` | `model.layer2[1].conv2` |

### Understanding the Problem

The debug output reveals that `layer2[1].relu` is causing overflow. Looking at our model:

```python
model.layer2[1].relu = UnstableReLU()  # This is the problem!
```

The `UnstableReLU` uses `torch.exp(x)` instead of standard ReLU, which causes exponential growth and overflow. The fix is to use the standard ReLU instead — or, if you really need exponential scaling, clamp the input first.

## Part 5: Fix the Problematic Layer

Now that we've identified the issue, let's fix it by using the standard ReLU. We do this simply by **not** entering the `patched_resnet18()` context:

```python
# fixed_finetuning.py (excerpt)
lightly_train.train_image_classification(
    out="out/debugging_fixed",
    data={ ... same dataset ... },
    model="torchvision/resnet18",  # Same model string as the broken run.
    # ... no patched_resnet18() here, so the standard ReLU is used.
)
```

```bash
python fixed_finetuning.py
```

The console output is back to the healthy gradient norm progression from Part 3:

```
Train Step  1/10 | Epoch 0 | train_loss: 2.2970 | lr: 0.00020000 | grad_norm: 4.6075
...
Train Step 10/10 | Epoch 0 | train_loss: 2.5596 | lr: 0.00000100 | grad_norm: 7.1247
```

Gradient norms remain stable, training completes successfully, and the comparison plot now has 10 data points in both runs.

## Part 6: Complete Debugging Workflow Summary

### Decision Tree: When to Use Each Tool

```
Training Issues?
│
├─ Look at console gradient norms (always on, no setup)
│  ├─ All `nan` from step 1 → catastrophic overflow
│  │  → Enable DebugUnderflowOverflow to find the layer
│  ├─ Step 1 normal, steps 2+ `nan` → overflow in backward pass
│  │  → Enable DebugUnderflowOverflow to find the layer
│  ├─ Growing rapidly (last/first > 100x) → exploding gradients
│  │  → Lower LR, add gradient clipping
│  ├─ Consistently decreasing to near zero → vanishing gradients
│  │  → Check architecture, add normalization
│  └─ Stable in a healthy range → training is fine
│
└─ Need the exact problematic layer?
   └─ Enable DebugUnderflowOverflow
      ├─ Identifies the module whose output went `inf`/`nan`
      ├─ Dumps last `max_frames_to_save` frames for context
      └─ Aborts training with a clear `ValueError`
```

### Quick Reference: Common Instability Patterns

| Symptom | Console gradient norm | Likely cause | Action |
| --- | --- | --- | --- |
| Step 1 `nan`, or step 1 normal then step 2+ `nan` | `nan` everywhere | Single layer overflows in fp16 | `DebugUnderflowOverflow` → fix the layer |
| Training unstable, no errors | Exponential growth (last/first > 100x) | Exploding gradients | Lower LR, add gradient clipping |
| Training stuck, high loss | Near zero, decreasing | Vanishing gradients | Check architecture, add normalization |
| Mixed precision issues | Random spikes | Layer instability | Check custom layers, force float32 |

### Key Takeaways

1. **Gradient norm logging** is your first diagnostic — always check it. `nan` gradient norms are a clear signal of numerical failure.
2. **`DebugUnderflowOverflow`** identifies the specific problematic layer when you need to know exactly which module is at fault. It's the only way to distinguish "the model overflowed at layer X" from "the model overflowed somewhere".
3. **Fix the root cause** — don't just mask symptoms with lower learning rates. If `DebugUnderflowOverflow` points to `layer2[1].relu`, replace that activation with a stable one.
4. **Test your fixes** — compare gradient norms before and after. A healthy run should produce finite gradient norms for the entire run.

## Advanced Usage

### PyTorch Autocast and `torch.exp`

A subtle gotcha worth knowing: PyTorch's autocast keeps `torch.exp` in float32 by default, regardless of the global precision setting. Without the explicit `.half()` cast inside `UnstableReLU`, the layer would **not** overflow in mixed precision training, even with `precision="16-mixed"`. This is a real-world issue when porting models across frameworks — custom ops that "should" overflow in fp16 sometimes don't, because autocast silently promotes them to fp32.

### `scale` Parameter

`UnstableReLU` accepts a `scale` argument:

```python
UnstableReLU(scale=1.0)  # Overflows immediately for activations > ~11
UnstableReLU(scale=0.8)  # "Step 1 normal, step 2+ NaN" — good for gradient-norm demo
UnstableReLU(scale=0.5)  # Too mild to show instability with small activations
```

Use `scale=1.0` when you want `DebugUnderflowOverflow` to abort immediately and dump the frame context. Use smaller values when you want to observe the gradient-norm degradation pattern.

### Trace Mode for Investigation

For detailed investigation without aborting training, use trace mode:

```python
debug_args={
    "underflow_overflow": {
        "enabled": True,
        "trace_batch_nums": [10, 20, 30],  # Trace specific steps
        "max_frames_to_save": 21,
    }
}
```

Trace mode disables detection and just dumps the full forward frames at the specified batch numbers. Useful for fast-forwarding to a known-bad region.

### Combined Debugging

Gradient-norm logging is always on by default. You can combine it with `DebugUnderflowOverflow` for a complete picture:

```python
lightly_train.train_image_classification(
    out="out/comprehensive_debugging",
    data=...,
    model="torchvision/resnet18",
    debug_args={
        "underflow_overflow": {
            "enabled": True,
            "max_frames_to_save": 21,
        }
    },
    # Gradient norm logging is always enabled by default.
)
```

## Additional Resources

For more information on debugging tools:

- **LightlyTrain debugging reference**: `debug_args` documentation in the LightlyTrain API reference.
- **Gradient-norm interpretation**: The console `grad_norm` field is the L2 norm of all trainable parameters' gradients after the clipping step.
- **`DebugUnderflowOverflow` internals**: Vendored from HuggingFace's `transformers.debug_utils` — see the source at `src/lightly_train/_debug/huggingface_debug_utils.py` for the exact frame-dump format.

This tutorial has shown you how to use LightlyTrain's debugging tools to identify and fix numerical instability issues during fine-tuning. The same workflow applies to your own models and debugging scenarios.