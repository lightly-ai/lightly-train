# Concept: data-dependent settings and layers

Status: draft for discussion, not planned. Based on `main` at a1c6c6c2 (with #983). PR #951 is still open and not
included. Related: #1018, the class-head refactor after #983.

## Short answer

These are two problems, and marking layers solves only one:

1. **Settings that must agree:** channels, mean/std, classes, image size, how images are loaded. They are worked out in
   about 30 copied blocks and partly guessed again at predict and export. Most of #1018 comes from here. A layer
   marker can't fix this.
2. **Weights that must be adapted:** class heads, the first conv or patch embed, positional embeddings. There are 7
   near-identical class-head hooks, plus channel and positional-embedding handling that some backbones have and others
   don't.

A fix that covers both needs three parts: one settings object, data-dependent layers handled by one shared loader, and
transforms that declare their requirements.

## What we found

Items marked "(not run)" were read from the code and not reproduced.

### What the checkpoint already stores

- **Task checkpoints:** `model_class_path` and `model_init_args` (`_train_task_state.py:29-30`). `model_init_args`
  holds the task model's constructor arguments, collected with `init_args=locals()` in `TaskModel.__init__`
  (`_task_models/task_model.py:39`).
- **Coverage of the planned `DataSpec` fields:**

  | Field | In `model_init_args`? |
  |---|---|
  | `classes` | yes |
  | `image_size` | yes |
  | `normalize` | yes, as `image_normalize`; can be `None` |
  | `model_channels` | only indirectly: `len(mean)`, or `backbone_args["in_chans"]` if the user set it. The two can disagree (#1016). |
  | `ignore_index` | only in the 4 semantic segmentation models |
  | `source_channels`, `channel_select` / `channel_drop`, image load mode | no |

- **The saved values are already reused on fine-tune.** `load_checkpoint` returns the checkpoint's `model_init_args`,
  and `train_task.py:1381` builds the new run from them. The transforms' `resolve_auto` reads `image_size` and
  `image_normalize` from them (e.g. `dinov3_eomt_instance_segmentation/transforms.py:106,115`).
- **Depth estimation** takes none of these fields as init arguments.
- **Pretraining checkpoints** store only `normalize_args` (`_checkpoint.py:37`).

So most of `DataSpec` is already saved, just untyped and mixed with architecture settings (`num_queries`,
`backbone_args`). What's missing is the channel information.

### Settings

- **The model's channel count is never stored.** Task models compute it as `len(image_normalize["mean"])`, in 23
  places, e.g. `_commands/export_task.py:140`. The checkpoint stores no `num_channels`, no on-disk channel count, no
  `channel_drop`, and no image load mode.
- **`num_channels` means two things.** It is used both for "channels on disk" (decides the image load mode) and
  "channels the model sees" (after `channel_drop`).
- **Copy-paste:**
  - The `"auto"` resolution blocks for `normalize`, `num_channels` and `image_size` are repeated ~28 times, twice per
    task transforms file.
  - These blocks handle `normalize=None` in 3 different ways:
    - EoMT and classification: becomes `NormalizeArgs()`.
    - LT-DETR: stays `None`.
    - PicoDet: a `"none"` sentinel that means ImageNet stats.
  - Mean/std are repeated to the channel count in 4 places (`_transforms/transform.py:333`,
    `image_classification_transform.py:89`, `eomt_transforms/utils.py:42`, `ltdetr_instance_segmentation/transforms.py:137`).
  - `_init_image_mode` exists in 4 dataset classes; #951 adds a 5th.
- **`channel_drop` silently loses data.** With keep=3 on 4-channel data, the images load as RGB. `ChannelDrop` then
  sees 3 channels and returns early (`_transforms/channel_drop.py:75`), so the 4th channel is lost with no error.
  With keep=2, the val transform loads 4 channels but normalizes them with 2 stats and should crash (not run).
- **Model and transform can disagree:**
  - LT-DETR builds the backbone with `len(mean)` channels but checks input against `_expected_input_channels`
    (`ltdetr_object_detection/task_model.py:205-225`).
  - `backbone_args["in_chans"]=4` with 3-value normalize stats trains, but `predict()` and ONNX export crash (#1016).
  - With `normalize=None`, LT-DETR always builds a 3-channel model.
- **Training and predict load images differently.** Training converts grayscale and RGBA images to RGB, while predict
  loads files unchanged (`_data/file_helpers.py:156-199`). An RGBA PNG at predict should fail with EoMT or
  classification (not run). Only LT-DETR checks the channel count, and rejects the image with a clear error.
- **Pretraining and fine-tuning are disconnected.** A pretraining checkpoint stores only `normalize_args`. A 4-channel
  pretrained backbone is fine-tuned as a 3-channel model by default, and the patch-embed hook silently slices its
  weights (not run).

### Weights

| Adaptation | Where | Policy |
|---|---|---|
| Patch embed ↔ channels | DINOv2/v3 ViT `patch_embed.py` → `_model_helpers.py:21` | slice / repeat, no rescale |
| `pos_embed` ↔ image size | DINOv2 `vision_transformer.py:238` → `_model_helpers.py:62` | bicubic; silently skips non-square grids |
| Class heads ↔ classes | 7 copies: `object_detection_components/hooks.py`, `eomt/hooks.py`, `pico_head.py`, `image_classification`, both multihead models | reinit when the class count differs |
| EoMT `queries` | `eomt/hooks.py:19` | truncate / repeat |
| `criterion.empty_weight` | `eomt/hooks.py:94` | always overwritten, even when shapes match |
| Patch-size resample | DINOv3 EoMT semantic segmentation `train_model.py:586` | train-model override, only this family |

Problems:

- **Same class count, different classes:** the old head weights are kept silently (e.g. COCO's 80 classes loaded onto
  a custom 80-class dataset). All hooks compare only the class count, never the class names, which are stored in
  `init_args`.
- **Linear semantic segmentation has no class-head hook** (`linear_semantic_segmentation/task_model.py:151`), so
  changing its classes should crash on load (not run).
- **Coverage is uneven (not run):**
  - DINOv3 ConvNeXt has no channel hook and loads with `strict=True`, so pretrained weights should fail to load when
    the channel count isn't 3.
  - timm and torchvision weights loaded through `backbone_weights` should fail the same way.
  - ECViT supports only 3 channels until #951.
- **PicoDet with `share_cls_reg=True`:** one conv outputs both class scores and box regression. A class change
  reinitializes the whole conv, so the pretrained box-regression weights are lost.
- **Per-layer hooks can't do this job.** A state-dict hook sees only the incoming tensors, not the checkpoint's saved
  settings. It can't tell "same shape, different classes" apart from a valid match.
- **But the caller already has those settings.** `finetune_from_checkpoint` (`train_task_helpers.py:1588`) holds
  `checkpoint["model_init_args"]`, including the old class names, and doesn't pass them on before calling
  `load_train_state_dict(strict=False)`. So "same class count, different classes" can be detected today, without
  `DataSpec`.
- **`finetune_from_checkpoint` loads with `strict=False` and only warns.** Multihead models with renamed heads, for
  example, keep their new heads randomly initialized, and the only sign is a warning.
- **Export can crash.** `export_task.py:140` reads `image_normalize["mean"]`, which should raise a `TypeError` when
  `image_normalize=None` (not run).

### Other libraries

- **timm:**
  - `pretrained_cfg` stores the model's `first_conv` and `classifier` layer paths, plus mean/std and input size.
  - One shared loader adapts the first conv (`adapt_input_conv`: sum for 1 channel, tile and scale by 3/N
    otherwise) and drops the classifier when `num_classes` differs.
  - Each model's `checkpoint_filter_fn` only renames keys and resamples positional embeddings.
  - Weakness: mean/std are not updated after `in_chans` changes.
- **Hugging Face:**
  - `ignore_mismatched_sizes` drops tensors whose shapes differ, then initializes only those.
  - Class names (`id2label`) live in `config.json`.
  - Preprocessing settings live in a separate `preprocessor_config.json` that nothing cross-checks against the model
    config, so the two drift apart. Avoid this.
- **torchvision / torchgeo:**
  - Weight enums bundle the weights, their metadata (categories, `in_chans`, bands) and the matching transforms.
  - torchgeo loads with `strict=False`, then asserts that only an allow-list of keys is missing.
- **ultralytics:** `_remap_cls_by_names` copies class-head rows by class name. This was the only library found that
  matches heads by class.
- **detectron2:** `pixel_mean`/`pixel_std` are registered as model buffers and applied in `forward`, so the stats are
  saved with the weights and included in the export.
- **PyTorch `get_extra_state`:** a poor place for these settings. They arrive only after the layers are already
  sized, are pickled, and don't work with `weights_only=True` or safetensors.
- **Channel-agnostic models** (DOFA, ChannelViT): remove the dependence of weight shapes on the channel count. This
  is a choice of model architecture, not something the loader can solve.

## Proposed concept

### Part 1: `DataSpec`, one settings object, worked out once and saved in the checkpoint

Mostly a cleanup: a typed, checked version of fields that `model_init_args` already stores, plus the missing channel
fields.

```python
class DataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Used by the model. Also required task model init args.
    classes: Dict[int, str]
    image_size: Tuple[int, int]       # patch-divisor already applied
    normalize: NormalizeArgs          # never None; validated: len(mean) == model_channels
    model_channels: int               # what the stem sees

    # Used by the dataset and predict only. Not init args.
    source_channels: int              # on disk → image load mode
    channel_select: ChannelSelect     # channel_drop / fixed selection, len == source_channels
    ignore_index: Optional[int]
```

- **Strict pydantic model, not `extra="allow"`:**
  - `extra="forbid"` catches typos such as `image_normalise`.
  - It keeps data settings apart from architecture settings (`num_queries`, `backbone_args`).
  - `model_dump(mode="json")` gives plain JSON, which works with `weights_only=True`.
- **Resolved once:** in `train_task`, from data args, transform args and the checkpoint's spec (if any), in a fixed
  order. This replaces the ~28 copied `"auto"` blocks.
- **Read by everything:**
  - The dataset (image load mode, replacing the `_init_image_mode` copies).
  - The transforms.
  - The task model constructor.
  - Predict, export and benchmark.
  - Nothing works these values out again.
- **Saved as plain JSON** in the checkpoint, next to `model_init_args`. Pretraining checkpoints save it too, which
  reconnects pretraining with fine-tuning.
- **One-way flow:** `train_task` fills the model fields of `model_init_args` from the spec
  (`model_init_args["classes"] = spec.classes`, …), never the reverse. The same fact is stored twice, but only one copy
  is ever written. On load, one check confirms that the two agree, which catches the `in_chans` vs `len(mean)`
  mismatch (#1016).
- **Old checkpoints:** `DataSpec.from_legacy(init_args)` reads `classes`, `image_size` and `image_normalize`, and
  guesses `model_channels = source_channels = len(mean)`.

#### Enforcing the shared init arguments

The model fields (`classes`, `image_size`, `image_normalize`, `model_channels`) must be init arguments of every task
model that uses them. Checked in `TaskModel.__init__`, not with an ABC, a Protocol or a test:

- **ABC:** `@abstractmethod` only checks that a method exists, not its arguments. Any `__init__` passes.
- **Protocol:** mypy ignores `__init__` when matching a class against a Protocol. Each model also has its own extra
  arguments, which would force `**kwargs` into one shared signature. `@runtime_checkable` checks names only.
- **Test:** works, but is separate from the code and easy to forget.
- **`TaskModel.__init__`:** every task model already passes `init_args=locals()` to it, so the base class sees every
  constructor argument. It can raise when a required field is missing and validate the model part of the spec
  (`len(mean) == model_channels`). This runs on every construction.

```python
missing = self.required_data_args - init_args.keys()
if missing:
    raise TypeError(f"{type(self).__name__} must accept init args {missing}.")
```

- **Per-model required set:** a class attribute `required_data_args`, because depth estimation has none of these
  fields and only the semantic segmentation models use `ignore_index`.
- **`data_spec` property:** implemented once on `TaskModel` from `init_args`, so it doesn't need to be abstract.

### Part 2: data-dependent layers are marked and handled by one shared loader

Each layer **declares** which `DataSpec` field it depends on. One loader then **adapts** it, with both the source spec
and the target spec in hand:

```python
class DataDependent(Protocol):
    depends_on: Literal["classes", "model_channels", "image_size"]
    def adapt_weights(self, state: Dict[str, Tensor], prefix: str,
                      src: DataSpec, dst: DataSpec) -> AdaptReport: ...
```

- **Policies are reusable:**
  - `ReinitOnClassChange`.
  - `RemapClassesByName`: copy class-head rows by class name, as ultralytics does.
  - `AdaptInputChannels`: repeat or sum, scaled by 3/N as in #1017.
  - `ResamplePosEmbed`.

  A layer is marked by wrapping it, or by a small registry on the task model, similar to timm's
  `pretrained_cfg["classifier"]`.
- **Exactly one loader:** `load_with_spec(model, ckpt)` runs the policies first, then calls
  `load_state_dict(strict=True)`. Loading stays strict, and any key that wasn't declared still raises an error. This
  avoids the silent shape-dropping that ultralytics and Hugging Face do.
- **The caller gets a report:** the loader returns which layers were reinitialized, remapped or resampled. Tests can
  check it and the log can print it.
- **Hooks still have a place,** but only for key renames and format conversion. That is the job timm gives its
  `checkpoint_filter_fn`.

### Part 3: transforms declare what they need, they don't adapt

- **No hooks needed.** Transforms don't hold weights, so there is nothing to adapt on load.
- **Each transform declares what it requires.** Examples: `requires_rgb = True` for hue/saturation; "any channel
  count" for mosaic once its canvas is fixed.
- **One resolver acts on those requirements.** It reads `DataSpec` and turns off the incompatible transforms with one
  log line. This replaces the 4 copies of the RGB-only check.
- **It also fixes the mosaic/`channel_drop` ordering bug,** because the resolver knows both `source_channels` and
  `model_channels`.

**Optional:** store mean/std as model buffers and normalize inside the model, as detectron2 does. The exported ONNX
model then takes raw pixels, and the stats can't disagree with the channel count.

## Suggested order

1. **Class-head refactor** (already planned). Write it as the first loader policy (`ReinitOnClassChange`) behind a
   small interface, not as a free-standing hook. Add linear semantic segmentation. Pass the checkpoint's
   `model_init_args` to it, so it can compare class names, not only counts.
2. **Add `DataSpec` and save it in checkpoints,** filled from today's `model_init_args` (`DataSpec.from_legacy`).
   Add the required-init-args check to `TaskModel.__init__`. Predict and export read the spec, which fixes the export
   `TypeError` and the #1016 crash.
3. **Split `source_channels` from `model_channels`.** This fixes the `channel_drop` bugs and the image-load-mode
   duplication.
4. **Move channel and positional-embedding adaptation into loader policies.** This covers ConvNeXt, timm and ECViT,
   and is the place for the #1017 scaling change, after a benchmark.
5. **Transform requirements resolver,** plus fixing mosaic for any channel count.

Steps 1 and 2 don't depend on each other. Steps 3 to 5 need step 2.

## Open questions

1. **Normalize inside the model?** It makes export much simpler, but changes how inference and export behave for
   every task model.
2. **Remap class-head rows by class name, or always re-initialize on any class change?** Remapping is friendlier when
   people add classes; re-initializing is simpler.
3. **Make `model_channels` an explicit init argument?** Then models stop computing `len(mean)` in 23 places, and the
   LT-DETR `normalize=None` → 3-channel bug goes away. But it touches every constructor. The alternative is to keep
   computing it and only check it against the spec on load. Leaning towards the explicit argument.
4. **Does the old checkpoint format need to keep loading forever,** or only through a one-time conversion step?
5. **PR #951:** merge it first, or fold its channel-adaptation pieces into step 4?
