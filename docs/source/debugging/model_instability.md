(model-instability-debugging)=

# Model Instability Debugging

Training instabilities — such as exploding or vanishing gradients, sudden loss spikes,
or numerical collapse to `NaN`/`inf` — can derail a run silently or abruptly. This page
collects the tools LightlyTrain provides to detect and diagnose these issues.

:::\{note} This section covers the debugging tools LightlyTrain ships for fine-tuning:
gradient norm logging (always on) and the on-demand `underflow_overflow` and
`nancapture` monitors.

:::

## Which Tool When

Start with the lightest signal and escalate as needed:

- **Trend of gradient magnitudes (always on):** `gradient_norm` is logged every step.
  See [Gradient Norm Logging](#gradient-norm-logging) below.
- **`NaN`/`inf` without an obvious culprit; localize the failing module:** see
  [Underflow/Overflow Detection](#underflow-overflow-detection).
- **Reproduce a sporadic bad step outside the live loop:** see
  [NaN/Inf Capture & Replay](#naninf-capture--replay).

`gradient_norm` is logged automatically. Enable the other two on demand via the
[`debug_args` setting](../settings/train_settings.md#debug). The full key lists and
output paths live in the same section.

## What Instability Looks Like

Common symptoms of an unstable run:

- The training loss spikes sharply or collapses to `NaN`/`inf`.
- The loss plateaus at a high value and never improves.
- The model stops learning partway through training (validation metrics flatten or
  regress).
- Training crashes with a numerical error during the forward or backward pass.

Not all of these mean instability — a high plateau can also be caused by a too low
learning rate or a data issue. Use the tools below to distinguish between them.

## Gradient Norm Logging

The total gradient norm is the single most useful signal for spotting exploding and
vanishing gradients. LightlyTrain logs it for every training step:

- `gradient_norm`: Total gradient norm computed after backpropagation, before the
  optimizer step. If gradient clipping is enabled (`gradient_clip_val > 0`) this is the
  pre-clipping norm; otherwise it is computed via an L2 norm. It is also shown in the
  console progress line as `grad_norm`.

It is written to all configured loggers (`metrics.jsonl`, TensorBoard, MLflow, Weights &
Biases) at the cadence set by
[`log_every_num_steps`](../settings/train_settings.md#log_every_num_steps).

### How to View the Gradient Norm

- **Console:** The progress line shows `grad_norm` for each logged training step.

- **TensorBoard:** Plot `gradient_norm` over training steps:

  ```bash
  tensorboard --logdir out/my_experiment
  ```

- **MLflow / Weights & Biases:** The `gradient_norm` metric is available under the same
  key. See [](../settings/train_settings.md) for how to enable these loggers.

### How to Interpret the Trend

Interpret the gradient norm as a trend over steps, not as an isolated value. Its
absolute scale depends on the model, dataset, and batch size, so there is no universal
"good" value. What matters is the shape:

- **Stable:** The norm fluctuates within a steady band across training.
- **Exploding gradients:** The norm grows rapidly, often by several orders of magnitude,
  and may precede a loss spike or a `NaN` collapse.
- **Vanishing gradients:** The norm shrinks toward zero and stays there, often
  accompanying a loss that no longer decreases.

A short-lived spike during warmup or learning-rate scheduling is usually normal. A
persistent upward or downward drift is the signal to act on.

### Common Next Actions

- **Exploding gradients:**
  - Lower the learning rate with [`model_args.lr`](../settings/train_settings.md).
  - Switch to a more stable precision, e.g. `precision="bf16-mixed"` or
    `precision="32-true"` (see [](../settings/train_settings.md)).
- **Vanishing gradients:**
  - Increase the learning rate, especially for small models (~10M parameters or fewer).
  - Check that the input normalization in `transform_args` matches your data
    distribution.
- **NaN/inf collapse:** Re-run from the latest checkpoint. If it reproduces, switch to
  `precision="32-true"` to isolate whether the instability is caused by
  reduced-precision arithmetic. For sporadic failures, escalate to
  [Underflow/Overflow Detection](#underflow-overflow-detection) to localize the failing
  module, then [NaN/Inf Capture & Replay](#naninf-capture--replay) to reproduce the bad
  step offline.

See the FAQ entry on [improving model performance](../faq.md) for broader guidance on
stable training.

(underflow-overflow-detection)=

## Underflow/Overflow Detection

When the gradient norm chart looks fine but forward passes start producing `NaN`/`inf`,
knowing *which module* first went bad shrinks the search dramatically. The
`underflow_overflow` monitor attaches forward hooks to every model module and reports
the absolute min/max of every weight, input, and output. Training aborts as soon as any
non-finite value is detected, and the last several forward frames are written to the
report so the failing module is straightforward to spot.

Enable it on demand when you suspect reduced-precision arithmetic (a specific task or
model occasionally collapses to `NaN`):

```python
debug_args={"underflow_overflow": {"enabled": True}}
```

The full key list (`max_frames_to_save`, `trace_batch_nums`, `abort_after_batch_num`) is
in the [underflow/overflow reference](../settings/train_settings.md#underflow_overflow).
Output is written per rank to `out/debug/underflow_overflow_rank{rank}.log` — the module
with the first non-finite value is where to look.

```{warning}
This tool significantly slows training — measured at roughly **3×** slower on
typical fine-tuning workloads (it runs an absolute `min`/`max` reduction on
every weight, input and output on each forward). Disable it once you have a
report. It also cannot be combined with `torch_compile_args={"disable": False}`;
see [Compile settings](../settings/train_settings.md#compilation).
```

(naninf-capture--replay)=

## NaN/Inf Capture & Replay

`gradient_norm` shows the trend; `underflow_overflow` shows the failing module. When
both still leave you unable to reproduce a sporadic `NaN`, `nancapture` snapshots the
failing step so you can replay it offline.

When enabled, the monitor scans parameter gradients for `NaN`/`Inf` after each
gradient-accumulation step (before the optimizer step). On detection it writes a
self-contained capture to `out/debug/nan_capture/rank{rank}/nan_capture.pt` holding the
model state, the step's microbatches, RNG state, and metadata — then aborts training by
raising `NaNDetectedError`.

Reproduce the failure without re-running training:

```python
from lightly_train._debug.nan_capture import load_nan_capture

cap = load_nan_capture("out/my_run/debug/nan_capture/rank0")
result = cap.replay()
print(result.reproduced, result.nan_param_names)
```

```{warning}
`nancapture` adds non-trivial per-step overhead — every step clones each
microbatch to CPU, snapshots the RNG, and scans parameter gradients for
`NaN`/`Inf`, plus a `torch.save` to disk on detection. Expect training to be
measurably slower (often **2–3×**) while the monitor is enabled. Disable it as
soon as you have the capture you need.
```

The replay reconstructs the `TrainModel`, restores the saved microbatches and RNG state,
and re-runs the triggering forward+backward pass; it stops before the optimizer step
(the corruption path that the training loop never reached). For mixed-precision
failures, pass a `Fabric` matching the captured run's precision: `cap.replay(fabric=f)`.

After diagnosing and fixing the bug, restart with `resume_interrupted=True`: the on-disk
`checkpoints/last.ckpt` is healthy because the capture raises before the optimizer step
and the per-step checkpoint save, so neither the bad gradient nor any bad optimizer
state is ever persisted. See the
[nancapture reference](../settings/train_settings.md#nancapture).
