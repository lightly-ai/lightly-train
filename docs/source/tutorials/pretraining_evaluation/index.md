(tutorials-pretraining-evaluation)=

# Evaluating Pretraining on Your Own Detection Data

So you've got a large pile of unlabeled images, a smaller set of annotated ones, and one
question: is pretraining on the unlabeled images actually going to improve your
detector? This guide shows how to set up that comparison so the answer is trustworthy:
which configurations to compare, what to hold fixed across runs, and what to record
before interpreting the numbers.

It builds directly on the
{ref}`Object Detection with Ultralytics' YOLOv26 <tutorials-yolo>` tutorial, which walks
through pretraining and fine-tuning end to end. Here we only cover the *evaluation
protocol* around it. The same principles apply if you fine-tune with LightlyTrain's own
{ref}`object detection models <object-detection>` instead.

```{note}
Pretraining is not guaranteed to beat every baseline on every dataset. Whether it
helps depends on factors like the domain shift of your data, the dataset size, and
the model size. The goal of this guide is not to promise an improvement — it is to
make sure that whatever result you measure is real.
```

```{warning}
Using Ultralytics models might require a commercial Ultralytics license. See the
[Ultralytics website](https://www.ultralytics.com/license) for more information.
```

## Prerequisites: A Realistic Starting Point

In practice you usually start with two directories: one with unlabeled images and one
with annotated images. If that is your situation, you are already set — skip ahead to
the next section.

To make this tutorial reproducible, we simulate that starting point from the PASCAL VOC
dataset (downloaded as in the {ref}`YOLOv26 tutorial <tutorials-yolo>`): we keep the
annotations for a small, randomly chosen part of the training images and treat the rest
as unlabeled. This is *only* scaffolding — with your own data these two directories
already exist.

From there we do what you would do with your own annotated directory: hold out part of
it for evaluation, and leave the rest for fine-tuning.

```python
import random
import shutil
from pathlib import Path

from ultralytics import settings

voc_dir = Path(settings["datasets_dir"]) / "VOC"
voc_train_dir = voc_dir / "images" / "train2012"
image_paths = sorted(voc_train_dir.glob("*.jpg"))
if not image_paths:
    raise FileNotFoundError(f"No images in {voc_train_dir}. Download VOC first.")

# Scaffolding only: pretend most images were never annotated.
random.Random(42).shuffle(image_paths)
annotated_paths, unlabeled_paths = image_paths[:1000], image_paths[1000:]

# The split you would also do on your own annotated directory.
train_paths, val_paths = annotated_paths[:800], annotated_paths[800:]


def copy_annotated(paths, out_dir):
    for subdir in ["images", "labels"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    for image_path in paths:
        label_path = voc_dir / "labels" / "train2012" / f"{image_path.stem}.txt"
        shutil.copy(image_path, out_dir / "images")
        if label_path.exists():  # An image without a label file counts as background.
            shutil.copy(label_path, out_dir / "labels")


copy_annotated(train_paths, Path("my_dataset/train"))
copy_annotated(val_paths, Path("my_valset"))

unlabeled_dir = Path("my_dataset/unlabeled")
unlabeled_dir.mkdir(parents=True, exist_ok=True)
for image_path in unlabeled_paths:
    shutil.copy(image_path, unlabeled_dir)
```

That leaves two top-level directories:

```text
my_dataset/          # everything pretraining is allowed to see
├── unlabeled/       # 4717 images, no annotations
└── train/           # 800 annotated images for fine-tuning

my_valset/           # 200 annotated images, held out for evaluation
```

Two details matter here, and they apply to your own data as well:

- **The splits are seeded and persisted.** Sorting before shuffling makes the result
  independent of filesystem order, so the same seed gives the same split on any machine.
  Every configuration you compare later must fine-tune on the *exact same* images and be
  evaluated on the *exact same* held-out set — never re-sample per run. If you do change
  the seed, delete `my_dataset/` and `my_valset/` first: files from the previous split
  are not cleaned up, and the two sets would end up overlapping.
- **The held-out set lives outside the pretraining directory.** `my_valset/` is
  deliberately not inside `my_dataset/`, so pointing pretraining at `my_dataset/` cannot
  reach it even by accident. Hold out as many images as you can afford — 200 keeps this
  example quick, but a small held-out set makes small differences hard to measure.

The dataset YAML for fine-tuning then references both, and stays identical across all
compared runs:

```yaml
# my_dataset.yaml
path: my_dataset/train
train: images
val: ../../my_valset/images
names: [aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
    diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
    tvmonitor]
```

## Choose What You're Comparing

Decide — and write down — which configurations you are comparing *before* training
anything. For YOLO-style detectors there are four meaningful starting points. Note which
tool each model string is passed to:

| Configuration              | `lightly_train.pretrain(model=…)` | `YOLO(…)` for fine-tuning |
| -------------------------- | --------------------------------- | ------------------------- |
| Random initialization      | not used                          | `"yolo26s.yaml"`          |
| COCO-pretrained weights    | not used                          | `"yolo26s.pt"`            |
| Pretraining from scratch   | `"ultralytics/yolo26s.yaml"`      | the exported checkpoint   |
| Pretraining on top of COCO | `"ultralytics/yolo26s.pt"`        | the exported checkpoint   |

The first two are baselines with no LightlyTrain step. The last two both pretrain on
your unlabeled images and differ only in what that pretraining starts from. We recommend
including the fourth: pretraining does not have to replace COCO weights, it can build on
them. Off-the-shelf initialization and pretraining on your own data are complementary,
not mutually exclusive.

```{warning}
Watch out for the head-initialization trap when comparing *COCO-pretrained weights*
against *pretraining from scratch*: the pretrained model has a strong *backbone* but a
randomly initialized *detection head*, while the off-the-shelf COCO checkpoint comes
with a fully trained head. The head accounts for a significant portion of the model's
parameters, so that comparison puts the pretrained model at a disadvantage that has
nothing to do with the quality of the pretraining. Pretraining on top of the COCO
weights keeps the head and avoids the imbalance.
```

## Draw the Data Boundary

Which images may enter the unlabeled pretraining pool?

- **Your labeled training images: yes.** Pretraining works best when it sees the full
  data distribution, and this will make your model better.
- **Your held-out images: never.** Including them leads to data leakage: the evaluation
  is no longer measuring generalization. Treat the pretraining pool like a training
  split — any image in it must not be used for evaluation.
- **Additional unlabeled images: yes.** The more the better, as long as they come from
  the same domain.

The layout from the prerequisites enforces this by construction: point pretraining at
`my_dataset/` and it sees the unlabeled images plus the fine-tuning training images,
nothing else. LightlyTrain walks nested folders on its own and ignores label files.

See the {ref}`FAQ <faq>` entries *"Can I train on labeled images?"* and *"How much data
do I need?"* for the underlying rules and recommended dataset sizes.

## Match the Training Budget and Run

The compared fine-tuning runs must differ in exactly one thing: the initialization. Keep
everything else identical across runs:

- Number of epochs (or steps) and batch size
- Learning rate and schedule
- Image size and augmentations
- Model architecture and size
- The labeled training data and the held-out set
- The `seed`

The learning rate deserves special attention: it usually has the largest effect of all
hyperparameters, and the best learning rate for a pretrained model can differ from the
best one for a randomly initialized model. If you tune it, tune it for each
configuration separately — and report what you used.

Three more rules that frequently decide whether the comparison is meaningful:

- **Don't starve the pretraining step.** A pretraining run that is too short will show
  no benefit — that is a property of the budget, not of pretraining. See the
  {ref}`recommended epochs and batch sizes <methods-distillation>` before concluding
  anything from a quick test run.
- **Evaluate every run on the identical, full held-out set.** Even when fine-tuning on a
  small labeled subset, validation uses the complete held-out set — never a shrunken
  one.
- **Use the same checkpoint-selection rule for every run.** For example, always report
  the best validation mAP50-95. Comparing the best epoch of one run against the final
  epoch of another silently biases the result.

The commands are the same as in the {ref}`YOLOv26 tutorial <tutorials-yolo>`, wired up
so that only the starting weights change between runs:

```python
from ultralytics import YOLO

import lightly_train

# Pretraining — only for the two pretrained configurations.
lightly_train.pretrain(
    out="out/pretrained_on_coco",
    model="ultralytics/yolo26s.pt",  # "ultralytics/yolo26s.yaml" starts from random
    data="my_dataset",
    epochs=100,
)

# Fine-tuning — identical for every run except the weights and the run name.
YOLO("out/pretrained_on_coco/exported_models/exported_last.pt").train(
    data="my_dataset.yaml",
    epochs=50,
    seed=0,
    project="logs/comparison",
    name="pretrained_on_coco",
)
```

Repeat the fine-tuning block for the other three configurations, changing only the
weights it starts from — `"yolo26s.yaml"`, `"yolo26s.pt"`, or the checkpoint exported by
the from-scratch pretraining run — and the `name`. The `seed` argument is documented in
the {ref}`Train Settings <train-settings>`, and in the
{ref}`Pretrain Settings <pretrain-settings>` for the pretraining run.

## Report Results

Record enough configuration alongside the metric that someone else — or you, three weeks
later — can interpret the comparison. A minimal results table:

| Run                     | Initialization             | Pretraining data / epochs | Fine-tuning data / epochs | Seed | mAP50-95 |
| ----------------------- | -------------------------- | ------------------------- | ------------------------- | ---- | -------- |
| random_init             | random                     | —                         | 800 images / 50           | 0    | …        |
| coco                    | COCO weights               | —                         | 800 images / 50           | 0    | …        |
| pretrained_from_scratch | LightlyTrain (from random) | my_dataset / 100          | 800 images / 50           | 0    | …        |
| pretrained_on_coco      | LightlyTrain (on COCO)     | my_dataset / 100          | 800 images / 50           | 0    | …        |

For Ultralytics fine-tuning the metric comes from the `results.csv` of each run, as
shown in the {ref}`YOLOv26 tutorial <tutorials-yolo>`. If you fine-tune with
LightlyTrain's `train_object_detection` instead, the
{ref}`benchmark command <quick-start-object-detection>` (in beta) evaluates a checkpoint
on a validation split and writes a report with the run configuration included.

## Interpret the Result

A few things to keep in mind before drawing conclusions:

- **One seed is one sample.** If the labeled subset is small or the difference between
  configurations is small, repeat the fine-tuning with 2–3 different seeds and report
  the spread. A gap that is smaller than the run-to-run variation is not a conclusion.
- **A null result is information, not failure.** If pretraining did not help, that is a
  valid, reportable outcome for your dataset and budget — it does not mean the
  comparison was wasted or that pretraining never helps.
- **If you want to dig further**, these are the levers that most often change the
  outcome:
  - Pretrain for longer — see the
    {ref}`recommended epochs for distillation <methods-distillation>`.
  - Pretrain on top of the COCO weights if you started from scratch.
  - Use a larger model; small models tend to profit less from pretraining.
  - Revisit the learning rate.

And if results stay ambiguous: the fine-tuning comparison you just ran *is* the ground
truth. Proxy metrics can speed up iteration, but the best way of truly knowing whether
pretraining helps on your data is exactly what you did — the whole fine-tuning.

## Next Steps

- Repeat the comparison with a different pretraining method — see
  {ref}`Methods <methods>` for the available options.
- Run the same protocol against LightlyTrain's own
  {ref}`LTDETR object detection models <object-detection>`, which ship with
  COCO-pretrained checkpoints.
- Revisit the {ref}`FAQ <faq>` for dataset size recommendations once you know how much
  your unlabeled data is worth.

Happy (fair) experimenting! 🔬
