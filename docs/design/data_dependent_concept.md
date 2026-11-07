# Concept: data-dependent settings and layers

Status: draft, decisions from the #1019 review folded in. Based on `main` at a1c6c6c2
(with #983). PR #951 is still open and not included; this work builds on top of it.
Related: #1016, #1018, #1015.

## Short answer

Three problems:

1. **Settings that must agree:** channels, mean/std, classes, image size. They are
   worked out in about 30 copied blocks and partly guessed again at predict and export.
   Most of #1018 comes from here.
1. **Weights that must be adapted to new data:** class heads, the first conv or patch
   embed, positional embeddings. There are 7 near-identical class-head hooks, plus
   channel and positional-embedding handling that some backbones have and others don't.
1. **Checkpoints have no format version.** Backwards compatibility is handled by hooks
   spread over the model code, and `None` values are allowed only so that old
   checkpoints keep loading.

The fix has five parts:

- **Migrations:** one versioned chain that upgrades every old checkpoint to the current
  format.
- **`DataSpec`:** one settings object, worked out once and saved in the checkpoint.
- **Shared loader with policies:** adapts data-dependent layers.
- **Transform requirements:** transforms declare what they need.
- **`EmbeddingTaskModel`:** gives pretraining the same checkpoint format as fine-tuning.

Principle: **fail hard on anything not supported.** Uneven coverage across models is
fine, as long as the limits are clear and unsupported cases raise an error instead of
silently doing something else.

## Decisions

From the #1019 review:

- **Normalize inside the model** (mean/std as buffers): no. Stats stay in the metadata.
- **Class-head adaptation:** remap rows by class name, not "reinit on any change".
  Classes kept from the checkpoint keep their weights, even when reordered.
- **`model_channels` is an explicit init argument** of every task model. No more
  `len(mean)`.
- **Old checkpoints** are upgraded by migrations. Breaking changes to the checkpoint
  format are allowed, as long as a migration covers them.
- **`DataSpec` fields:** `classes`, `image_size` and `ignore_index` are optional, so
  depth estimation and embedding models fit the same class. No subclasses for now.
- **`source_channels`** (channels on disk) stays in the training config, not in the
  saved spec. A saved model only knows `model_channels`.
- **Predict** fails when the image channel count differs from `model_channels`.
- **State-dict hooks:** replaced by migrations and loader policies. Exceptions are
  listed under Part 2.
- **PR #1015** is changed into the first loader policy, not closed.
- **`EmbeddingTaskModel`** is in scope. It predicts one pooled vector per image.
  `lightly_train.embed` is deprecated.
- **StrEnum over Literal** for internal code: new code, and code that touches an
  existing `Literal`, uses a `StrEnum`. This goes into the code guidelines.
- **`strict=False` with only a warning** on fine-tune loads is not wanted. Loading is
  strict.

## What we found

Items marked "(not run)" were read from the code and not reproduced.

### Checkpoint format

- **Task checkpoints store no version.** `CheckpointDict` (`_train_task_state.py:34`)
  and the exported `model_dict` (`train_task.py:1889`) have `model_class_path`,
  `model_init_args` and the weights, but no package or format version. Pretraining
  checkpoints do store `version` (`_checkpoint.py:34`).

- **Two task formats with different keys:**

  | File                               | Weights key                                  | Used by                                        |
  | ---------------------------------- | -------------------------------------------- | ---------------------------------------------- |
  | `checkpoints/last.ckpt`            | `train_model_state_dict` (+ optimizer state) | `resume_interrupted`, fine-tuning              |
  | `exported_models/exported_last.pt` | `train_model`                                | `load_model()`, predict, export, hosted models |

- **Hosted models** (`DOWNLOADABLE_MODEL_URL_AND_HASH`) are old `exported_last.pt`
  files. They can't be re-exported without changing their hash, so migrations must be
  able to upgrade them on load.

- **The 32 hook registrations are three different kinds:**

  | Kind                                         | Hooks                                                                                                                                              |
  | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Old format → new format (same model)         | `_remap_legacy_keys` (DINOv3 ConvNeXt wrapper, `DINOSTAs`)                                                                                         |
  | Data A → data B (fine-tuning)                | class heads (7 kinds), `queries_adjust_num_queries`, `criterion_empty_weight_reinit`, `patch_embed_adjust_input_channels`, `interpolate_pos_embed` |
  | Architecture config differs from the weights | `_pop_ignored_state_dict_keys` (hybrid encoder drops `downsample_convs.` for ViT configs)                                                          |

### What the checkpoint already stores

- `model_init_args` holds the task model's constructor arguments, collected with
  `init_args=locals()` in `TaskModel.__init__` (`_task_models/task_model.py:39`).

  | Field            | In `model_init_args`?                                                                                          |
  | ---------------- | -------------------------------------------------------------------------------------------------------------- |
  | `classes`        | yes                                                                                                            |
  | `image_size`     | yes                                                                                                            |
  | `normalize`      | yes, as `image_normalize`; can be `None`                                                                       |
  | `model_channels` | only indirectly: `len(mean)`, or `backbone_args["in_chans"]` if the user set it. The two can disagree (#1016). |
  | `ignore_index`   | only in the 4 semantic segmentation models, as `class_ignore_index`                                            |

- **`ignore_index` vs `ignore_classes`:** different things. `ignore_classes` is a
  dataset setting for all tasks that drops class IDs from the labels. `ignore_index` is
  the mask value for unlabeled pixels (default `-100`,
  `mask_semantic_segmentation_dataset.py:318`). Only semantic segmentation needs it, to
  map it back at predict.

- **Saved values are reused on fine-tune:** `train_task.py:1381` builds the new run from
  the checkpoint's `model_init_args`, and the transforms' `resolve_auto` reads
  `image_size` and `image_normalize` from them.

- **Depth estimation** takes none of these fields as init arguments today.

- **Pretraining checkpoints** store only `normalize_args` (`_checkpoint.py:37`). There
  is no task model through which a pretrained backbone can be loaded for predict.

### Settings

- **The model's channel count is never stored.** Task models compute it as
  `len(image_normalize["mean"])`, in 23 places, e.g. `_commands/export_task.py:140`.
- **`num_channels` means two things:** "channels on disk" (decides the image load mode)
  and "channels the model sees" (after `channel_drop`).
- **Copy-paste:**
  - The `"auto"` resolution blocks for `normalize`, `num_channels` and `image_size` are
    repeated ~28 times, twice per task transforms file.
  - `normalize=None` is handled in 3 different ways: EoMT and classification turn it
    into `NormalizeArgs()`, LT-DETR keeps `None`, and PicoDet uses a `"none"` sentinel
    that means ImageNet stats.
  - Mean/std are repeated to the channel count in 4 places
    (`_transforms/transform.py:333`, `image_classification_transform.py:89`,
    `eomt_transforms/utils.py:42`, `ltdetr_instance_segmentation/transforms.py:137`).
  - `_init_image_mode` exists in 4 dataset classes; #951 adds a 5th.
- **`channel_drop`:**
  - Only one strategy exists: drop `C - num_channels_keep` random channels, weighted by
    `weight_drop` (`_transforms/channel_drop.py`). Nothing selects channels at val or
    predict.
  - It was meant for pretraining, but all 13 task transforms accept it and
    `docs/source/settings/train_settings.md:753` documents it for fine-tuning.
  - It silently loses data: with keep=3 on 4-channel data, the images load as RGB,
    `ChannelDrop` sees 3 channels and returns early (`channel_drop.py:75`). With keep=2,
    the val transform loads 4 channels but normalizes them with 2 stats and should crash
    (not run).
- **Model and transform can disagree:**
  - LT-DETR builds the backbone with `len(mean)` channels but checks input against
    `_expected_input_channels` (`ltdetr_object_detection/task_model.py:205-225`).
  - `backbone_args["in_chans"]=4` with 3-value normalize stats trains, but `predict()`
    and ONNX export crash (#1016).
  - With `normalize=None`, LT-DETR always builds a 3-channel model.
- **Training and predict load images differently.** Training converts grayscale and RGBA
  images to RGB, while predict loads files unchanged (`_data/file_helpers.py:156-199`).
  Only the LT-DETR models check the channel count at predict. The others crash later or
  run on wrong input (not run).
- **Export can crash:** `export_task.py:140` reads `image_normalize["mean"]`, which
  should raise a `TypeError` when `image_normalize=None` (not run).

### Weights

| Adaptation               | Where                                                                                                                            | Policy                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Patch embed ↔ channels   | DINOv2/v3 ViT `patch_embed.py` → `_model_helpers.py:21`                                                                          | slice / repeat, no rescale                 |
| `pos_embed` ↔ image size | DINOv2 `vision_transformer.py:238` → `_model_helpers.py:62`                                                                      | bicubic; silently skips non-square grids   |
| Class heads ↔ classes    | 7 copies: `object_detection_components/hooks.py`, `eomt/hooks.py`, `pico_head.py`, `image_classification`, both multihead models | reinit when the class count differs        |
| EoMT `queries`           | `eomt/hooks.py:19`                                                                                                               | truncate / repeat                          |
| `criterion.empty_weight` | `eomt/hooks.py:94`                                                                                                               | always overwritten, even when shapes match |
| Patch-size resample      | DINOv3 EoMT semantic segmentation `train_model.py:586`                                                                           | train-model override, only this family     |

Problems:

- **Same class count, different classes:** the old head weights are kept silently (e.g.
  COCO's 80 classes loaded onto a custom 80-class dataset). All hooks compare only the
  class count, never the class names, which are stored in `init_args`.
- **Linear semantic segmentation has no class-head hook**
  (`linear_semantic_segmentation/task_model.py:151`), so changing its classes crashes on
  load. #1015 adds it.
- **Unsupported cases don't fail clearly (not run):**
  - DINOv3 ConvNeXt has no channel hook and loads with `strict=True`, so pretrained
    weights should fail to load with a size-mismatch error, not a clear "not supported".
  - timm and torchvision weights loaded through `backbone_weights` should fail the same
    way.
  - ECViT supports only 3 channels until #951.
- **PicoDet:** `share_cls_reg=True` is hard-coded
  (`picodet_object_detection/task_model.py:144,155`), so one conv outputs class scores
  and box regression together (`pico_head.py:350`). A class change reinitializes the
  whole conv and loses the pretrained box-regression weights. A remap policy fixes this:
  copy the box rows, remap the class rows.
- **Per-layer hooks can't see the checkpoint's settings.** A state-dict hook sees only
  tensors, so it can't tell "same shape, different classes" from a valid match. The
  caller can: `finetune_from_checkpoint` (`train_task_helpers.py:1588`) holds
  `checkpoint["model_init_args"]` but doesn't pass it on.
- **`finetune_from_checkpoint` loads with `strict=False` and only warns.** Multihead
  models with renamed heads, for example, keep their new heads randomly initialized, and
  the only sign is a warning.

### Other libraries

- **timm:** `pretrained_cfg` stores the `first_conv` and `classifier` layer paths, plus
  mean/std and input size. One shared loader adapts the first conv (`adapt_input_conv`:
  sum for 1 channel, tile and scale by 3/N otherwise) and drops the classifier when
  `num_classes` differs. Weakness: mean/std are not updated after `in_chans` changes.
- **Hugging Face:** `ignore_mismatched_sizes` drops tensors whose shapes differ.
  Preprocessing settings live in a separate `preprocessor_config.json` that nothing
  cross-checks against the model config, so the two drift apart. Avoid this.
- **torchvision / torchgeo:** weight enums bundle weights, metadata (categories,
  `in_chans`, bands) and transforms. torchgeo loads with `strict=False`, then asserts
  that only an allow-list of keys is missing.
- **ultralytics:** `_remap_cls_by_names` copies class-head rows by class name. The model
  for our remap policy.
- **Alembic:** each migration has an ID and a pointer to its parent; the database stores
  its current ID, and `upgrade` walks the chain. Random IDs plus merge revisions let
  parallel branches each add migrations.
- **PyTorch `get_extra_state`:** a poor place for these settings. They arrive only after
  the layers are already sized, are pickled, and don't work with `weights_only=True` or
  safetensors.

## Proposed concept

### Part 0: checkpoint format version and migrations

- **Integer format version:** every checkpoint stores `checkpoint_format_version: int`,
  separate from the package version, which changes on every release even when the format
  doesn't. Checkpoints without it are version 0.
- **Linear chain:** a list of steps `vN → vN+1`, run in order until the current version.
  - Simpler than Alembic: checkpoints only move forward, and all migrations live in one
    repo. Two PRs claiming the same number is a merge conflict that a test catches.
  - Upgrade only, no downgrade. A checkpoint newer than the installed package raises
    "please upgrade lightly-train".
- **One chain for both formats:** `last.ckpt` and `exported_last.pt` carry the version
  and run the same migrations on `model_class_path`, `model_init_args` and the state
  dict. Only the weights key differs.
- **Runs right after `torch.load`,** before the model is built. The model never resolves
  or guesses settings.
- **A migration changes the format, never the behavior.** After it runs, the model
  predicts exactly as before.
- **First migration (v0 → v1):**
  - The legacy key renames from `_remap_legacy_keys` (DINOv3 ConvNeXt wrapper,
    `DINOSTAs`).
  - Fill the `DataSpec` fields: `classes`, `image_size`, `normalize` from
    `model_init_args`. `normalize=None` becomes the value each model uses for `None`
    today, so behavior doesn't change.
  - `model_channels` from the stem weight shape, which is the real channel count even
    when `len(mean)` and `in_chans` disagree (#1016). This needs each model's stem key.
    Keep it simple: cover the known cases, and fix the rest when someone opens an issue.
- **Tests:** small fixture checkpoints in `tests/`, one per format version and model
  family. No downloads in unit tests or CI. Hosted models are checked by hand during
  development.

### Part 1: `DataSpec`, one settings object, worked out once and saved

```python
class DataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_channels: int
    normalize: NormalizeArgs           # validated: len(mean) == model_channels
    classes: Optional[Dict[int, str]]  # None for depth estimation and embeddings
    image_size: Optional[Tuple[int, int]]
    ignore_index: Optional[int]        # semantic segmentation only
```

- **Strict pydantic model:** `extra="forbid"` catches typos and keeps data settings
  apart from architecture settings (`num_queries`, `backbone_args`).
  `model_dump(mode="json")` gives plain JSON, which works with `weights_only=True`.
- **Optional fields, no subclasses** for now. A model that needs a field checks that it
  is set.
- **Resolved once:** in `train_task`, from data args, transform args and the
  checkpoint's spec (if any), in a fixed order. This replaces the ~28 copied `"auto"`
  blocks and the 3 ways of handling `normalize=None`.
- **Read by everything:** the dataset, the transforms, the task model constructor,
  predict, export and benchmark. Nothing works these values out again.
- **Saved as plain JSON** next to `model_init_args`, in both checkpoint formats.
- **One-way flow:** `train_task` fills the model fields of `model_init_args` from the
  spec, never the reverse. On load, one check confirms that the two agree.
- **Not in the spec:** `source_channels` and the image load mode. They describe the
  training data, not the model, and stay in the training config.

#### Enforcing the shared init arguments

- **Checked in `TaskModel.__init__`:** every task model passes `init_args=locals()`, so
  the base class sees every constructor argument. It builds a `DataSpec` from them and
  raises when a required field is missing or `len(mean) != model_channels`. This runs on
  every construction.
- **Why not an ABC, a Protocol or a test:** `@abstractmethod` doesn't check arguments.
  mypy ignores `__init__` when matching a Protocol. A test is separate from the code and
  easy to forget.
- **Required fields are shared:** `model_channels` and `normalize` for every task model.
  Optional fields are checked by the models that use them.
- **`data_spec` property:** abstract on `TaskModel` while the design settles. Each model
  returns its spec.

### Part 2: data-dependent layers are adapted by one shared loader

Each data-dependent layer **declares** which `DataSpec` field it depends on. One loader
**adapts** it, with both the source spec and the target spec in hand:

```python
class DataField(StrEnum):
    CLASSES = "classes"
    MODEL_CHANNELS = "model_channels"
    IMAGE_SIZE = "image_size"


class DataDependent(Protocol):
    depends_on: DataField

    def adapt_weights(
        self, state: Dict[str, Tensor], prefix: str, src: DataSpec, dst: DataSpec
    ) -> AdaptReport: ...
```

- **Policies are reusable:**
  - `RemapClassesByName`: copy class-head rows by class name, initialize new classes.
    Handles reordering. For PicoDet, the box-regression rows are copied unchanged.
  - `AdaptInputChannels`: repeat or sum, scaled by 3/N as in #1017.
  - `ResamplePosEmbed`.
  - EoMT `queries` and `criterion.empty_weight` become policies too.
- **Exactly one loader:** `load_with_spec(model, ckpt)` runs migrations, then the
  policies, then `load_state_dict(strict=True)`. Any key that no policy handled still
  raises an error.
- **The caller gets a report** of which layers were remapped, reinitialized or
  resampled. Tests check it, the log prints it.
- **Unsupported cases fail with a clear error,** e.g. a channel change on a backbone
  without `AdaptInputChannels`.
- **Hooks that stay:**
  - `patch_embed_adjust_input_channels` and `interpolate_pos_embed` in the vendored
    DINOv2/v3 code. They are needed to load upstream weights, which come with no spec.
    They must do nothing when shapes already match, so running them after a policy is
    harmless. Each gets a test for that.
  - `_pop_ignored_state_dict_keys` in the hybrid encoder: it handles an architecture
    config that leaves out modules, not data or checkpoint age.
- **All other hooks go:** class-head and EoMT hooks become policies;
  `_remap_legacy_keys` becomes the first migration.

### Part 3: transforms declare what they need, they don't adapt

- **No hooks needed.** Transforms hold no weights.
- **Each transform declares its requirements,** e.g. `requires_rgb = True` for
  hue/saturation, or "any channel count" for mosaic once its canvas is fixed.
- **One resolver** reads `DataSpec` and the training config, turns off incompatible
  transforms with one log line, and replaces the 4 copies of the RGB-only check.
- **It also fixes the mosaic/`channel_drop` ordering bug,** because it knows both
  `source_channels` (training config) and `model_channels` (spec).

### Part 4: `EmbeddingTaskModel`

- **A task model for pretrained backbones:** predicts one pooled vector per image, and
  saves and loads in the task checkpoint format with a `DataSpec`.
- **Pretraining saves one too.** This reconnects pretraining with fine-tuning: a
  4-channel pretrained backbone is fine-tuned as a 4-channel model, not silently sliced
  to 3.
  - Written as an extra file next to `exported_models/exported_last.pt`, which stays
    unchanged so existing scripts keep working.
  - Only for the final model at the end of pretraining, not for intermediate
    checkpoints. This matches when `exported_last.pt` is written today
    (`_commands/train.py:532`).
- **`image_size`** for embeddings is the global crop size of the pretraining method. The
  DINO and distillation transforms already have an `image_size` setting (default
  `(224, 224)`).
- **`lightly_train.embed` is deprecated** and removed in a future version (not yet set).
- **Not in scope:** moving the pretraining pipeline onto the fine-tuning pipeline.

## Suggested order

0. **Format version and migration runner.** Add `checkpoint_format_version` to both task
   formats, and the v0 → v1 migration with the legacy key renames. Ship this first,
   before anything else changes the format.
1. **Rework #1015 into the first policy,** `RemapClassesByName`, with the loader and its
   report. Includes linear semantic segmentation and the PicoDet box rows. Pass the
   checkpoint's class names, not only counts.
1. **Add `DataSpec`,** saved in both formats. Extend the first migration to fill it from
   `model_init_args` and the stem weight shape. Make `model_channels` an init argument.
   Predict and export read the spec, which fixes the export `TypeError` and the #1016
   crash. Predict fails on a channel mismatch.
1. **`EmbeddingTaskModel`,** pretraining saves it, `embed` deprecated.
1. **Move channel, positional-embedding, `queries` and `empty_weight` adaptation into
   policies.** Unsupported backbones (ConvNeXt, timm, torchvision) fail with a clear
   error. The #1017 scaling change goes here, after a benchmark. Builds on #951 for
   ECViT.
1. **Transform requirements resolver,** plus fixing mosaic for any channel count.
1. **`channel_drop` strategies:** replace the single random strategy with a proper
   strategy specification for how channels are dropped. Which strategies, and whether
   fine-tuning keeps `channel_drop`, is decided at this step.

Step 0 comes first. Steps 1 and 2 don't depend on each other. Steps 3 to 6 need step 2.

## Conventions

- Internal code uses `StrEnum` instead of `Literal`: in new code, and in code that
  touches an existing `Literal`. To be added to the code guidelines.

## Open questions

None at the moment. Deferred to their step:

- `channel_drop` strategy specification and whether fine-tuning keeps it (step 6).
- The version in which `lightly_train.embed` is removed.
