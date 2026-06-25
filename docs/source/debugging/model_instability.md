(model-instability-debugging)=

# Model Instability Debugging

Training instabilities — such as exploding or vanishing gradients, sudden loss spikes,
or numerical collapse to `NaN`/`inf` — can derail a run silently or abruptly. This page
collects the tools LightlyTrain provides to detect and diagnose these issues.

> [!NOTE] This section is growing. More debugging tools will be documented here as they
> are added. The first available tool is gradient norm logging.

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
  pre-clipping norm; otherwise it is computed via an unbounded (`inf`) norm. It is also
  shown in the console progress line as `grad_norm`.

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
  - Resume from the last good checkpoint using
    [`resume_interrupted`](../settings/train_settings.md) after changing settings.
- **Vanishing gradients:**
  - Increase the learning rate, especially for small models (~10M parameters or fewer).
  - Check that the input normalization in `transform_args` matches your data
    distribution.
- **NaN/inf collapse:** Re-run from the latest checkpoint. If it reproduces, switch to
  `precision="32-true"` to isolate whether the instability is caused by
  reduced-precision arithmetic.

See the FAQ entry on [improving model performance](../faq.md) for broader guidance on
stable training.
