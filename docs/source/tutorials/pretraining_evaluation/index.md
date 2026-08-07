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
annotations for a small, randomly chosen part of the images and treat the rest as
unlabeled. This split is *only* scaffolding — with your own data it already exists.

```python
import random
import shutil
from pathlib import Path

for subdir in ["labeled/images", "labeled/labels", "unlabeled"]:
    Path("my_dataset", subdir).mkdir(parents=True, exist_ok=True)

image_paths = sorted(Path("datasets/VOC/images/train2012").glob("*.jpg"))
random.Random(42).shuffle(image_paths)
labeled_paths, unlabeled_paths = image_paths[:1000], image_paths[1000:]

for image_path in labeled_paths:
    split_name = image_path.parent.name
    label_name = image_path.with_suffix(".txt").name
    label_path = image_path.parents[2] / "labels" / split_name / label_name
    shutil.copy(image_path, "my_dataset/labeled/images")
    shutil.copy(label_path, "my_dataset/labeled/labels")

for image_path in unlabeled_paths:
    shutil.copy(image_path, "my_dataset/unlabeled")
```

Two details matter here, and they apply to your own data as well:

- **The split is seeded and persisted.** Sorting before shuffling makes the result
  independent of filesystem order, so the same seed gives the same split on any machine.
  Every configuration you compare later must fine-tune on the *exact same* labeled
  images — never re-sample the subset per run.
- **Validation data stays outside.** `my_dataset/` contains training data only. The
  validation images (`val2012`) remain where they are and never enter the pretraining
  pool.

For fine-tuning, point a small dataset YAML at the labeled subset — and note that
validation always uses the full, original validation set (adjust paths to your setup):

```yaml
# my_dataset.yaml
path: my_dataset/labeled
train: images
val: ../../datasets/VOC/images/val2012 # the full val set, identical for every run
names: [aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
    diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train,
    tvmonitor]
```

## Choose What You're Comparing

Decide — and write down — which configurations you are comparing *before* training
anything. For YOLO-style detectors there are three meaningful starting points. Note
which tool each model string is passed to:

| Configuration              | Pretraining step (LightlyTrain)                               | Fine-tuning starts from (Ultralytics)              |
| -------------------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| Random initialization      | —                                                             | `YOLO("yolo26s.yaml")`                             |
| COCO-pretrained weights    | —                                                             | `YOLO("yolo26s.pt")`                               |
| Pretraining on top of COCO | `pretrain(model="ultralytics/yolo26s.pt", data="my_dataset")` | `YOLO("out/.../exported_models/exported_last.pt")` |

We recommend including the third configuration: pretraining does not have to replace
COCO weights, it can build on them. Off-the-shelf initialization and pretraining on your
own data are complementary, not mutually exclusive.

```{warning}
Watch out for the head-initialization trap: a model exported after LightlyTrain
pretraining has a pretrained *backbone* but a randomly initialized *detection head*,
while an off-the-shelf COCO checkpoint comes with a fully trained head. The head
accounts for a significant portion of the model's parameters, so comparing these two
directly puts the pretrained model at a disadvantage that has nothing to do with the
quality of the pretraining. Pretraining on top of the COCO weights avoids this
imbalance.
```

## Draw the Data Boundary

Which images may enter the unlabeled pretraining pool?

- **Your labeled training images: yes.** Pretraining works best when it sees the full
  data distribution, and this will make your model better.
- **Validation and test images: never.** Including them leads to data leakage: the
  evaluation is no longer measuring generalization. Treat the pretraining pool like a
  training split — any image in it must not be used for evaluation.
- **Additional unlabeled images: yes.** The more the better, as long as they come from
  the same domain.

With the directory layout from the prerequisites this boundary is enforced by
construction: point the pretraining at the whole `my_dataset` directory — LightlyTrain
finds images in nested folders on its own and ignores label files — and the validation
images can never leak in because they live outside of it.

See the {ref}`FAQ <faq>` entries *"Can I train on labeled images?"* and *"How much data
do I need?"* for the underlying rules and recommended dataset sizes.

## Match the Training Budget and Run

The compared fine-tuning runs must differ in exactly one thing: the initialization. Keep
everything else identical across runs:

- Number of epochs (or steps) and batch size
- Learning rate and schedule
- Image size and augmentations
- Model architecture and size
- The labeled training data and the validation data
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
- **Evaluate every run on the identical, full validation set.** Even when fine-tuning on
  a small labeled subset, validation uses the complete original validation split — never
  a shrunken one.
- **Use the same checkpoint-selection rule for every run.** For example, always report
  the best validation mAP50-95. Comparing the best epoch of one run against the final
  epoch of another silently biases the result.

The training commands themselves are the same as in the
{ref}`YOLOv26 tutorial <tutorials-yolo>`; just give each run a distinct name, e.g.
`name="from_scratch"`, `name="from_coco"`, and `name="pretrained_on_coco"`. If you use
LightlyTrain for fine-tuning, the `seed` argument is documented in the
{ref}`Train Settings <train-settings>` (and {ref}`Pretrain Settings <pretrain-settings>`
for the pretraining run).

## Report Results

Record enough configuration alongside the metric that someone else — or you, three weeks
later — can interpret the comparison. A minimal results table:

| Run                | Initialization      | Pretraining data / epochs | Fine-tuning data / epochs | Seed | mAP50-95 |
| ------------------ | ------------------- | ------------------------- | ------------------------- | ---- | -------- |
| from_scratch       | random              | —                         | 1000 images / 50          | 0    | …        |
| from_coco          | COCO weights        | —                         | 1000 images / 50          | 0    | …        |
| pretrained_on_coco | COCO + LightlyTrain | my_dataset / …            | 1000 images / 50          | 0    | …        |

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
- **If you want to dig further**, the levers that most often change the outcome are:
  training longer (see the
  {ref}`recommended epochs for distillation <methods-distillation>`), pretraining on top
  of the COCO weights if you started from scratch, using a larger model (small models
  tend to profit less from pretraining), and revisiting the learning rate.

And if results stay ambiguous: the fine-tuning comparison you just ran *is* the ground
truth. Proxy metrics can speed up iteration, but the best way of truly knowing whether
pretraining helps on your data is exactly what you did — the whole fine-tuning.

Happy (fair) experimenting! 🔬
