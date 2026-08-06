(tutorials-pretraining-evaluation)=

# Evaluating Pretraining on Your Own Detection Data

So you've pretrained a model on your own images — how do you tell whether it actually
helped? This guide shows how to set up that comparison so the answer is trustworthy:
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

## Choose What You're Comparing

Decide — and write down — which configurations you are comparing *before* training
anything. For YOLO-style detectors there are three meaningful starting points:

| Configuration              | How to get it                                                                                   | What it tells you                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Random initialization      | `model="ultralytics/yolo26s.yaml"`                                                              | Does pretraining beat no prior knowledge at all?           |
| COCO-pretrained weights    | `model="yolo26s.pt"` directly in fine-tuning                                                    | Does pretraining beat the standard off-the-shelf start?    |
| Pretraining on top of COCO | `model="ultralytics/yolo26s.pt"` in `lightly_train.pretrain`, then fine-tune the exported model | Does pretraining add value *on top of* the standard start? |

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

Before assembling the unlabeled pretraining dataset, decide which images are allowed in
it:

- **Fine-tuning training images: yes.** Include them — pretraining works best when it
  sees the full data distribution, and this will make your model better.
- **Validation and test images: never.** Including them leads to data leakage: the
  evaluation is no longer measuring generalization. Treat the unlabeled pretraining set
  like a training split — any image in it must not be used for evaluation.
- **Additional unlabeled images: yes.** The more the better, as long as they come from
  the same domain.

See the {ref}`FAQ <faq>` entries *"Can I train on labeled images?"* and *"How much data
do I need?"* for the underlying rules and recommended dataset sizes.

For a VOC-style layout like in the {ref}`YOLOv26 tutorial <tutorials-yolo>` this means:

```text
datasets/VOC
├── images
│   ├── train2007      # ✅ include in pretraining
│   ├── train2012      # ✅ include in pretraining
│   ├── val2007        # ❌ never include
│   ├── val2012        # ❌ never include
│   └── test2007       # ❌ never include
└── labels             # ignored by pretraining
```

## Create Reproducible Labeled Subsets

Skip this section if you fine-tune on your full labeled set. But a common experiment is
"how much does pretraining help when only 10% of my data is labeled?" — and for that
comparison to be valid, every configuration must fine-tune on the *exact same* images.
Do not re-sample a random subset per run.

Create the subset once, with a fixed seed, and persist the file list:

```python
import random
from pathlib import Path

image_paths = sorted(Path("datasets/VOC/images/train2012").glob("*.jpg"))
random.Random(42).shuffle(image_paths)
subset = image_paths[:1000]

Path("subset_1000_seed42.txt").write_text("\n".join(str(p) for p in subset))
```

Sorting before shuffling makes the result independent of filesystem order, so the same
seed gives the same subset on any machine. Reuse this file for every configuration —
most fine-tuning frameworks accept a text file with image paths as a training split
(e.g. in an Ultralytics dataset YAML).

## Match the Training Budget and Run

The compared fine-tuning runs must differ in exactly one thing: the initialization. Keep
everything else identical across runs:

- Number of epochs (or steps) and batch size
- Learning rate and schedule
- Image size and augmentations
- Model architecture and size
- The labeled training data (see the previous section)
- The `seed`

The learning rate deserves special attention: it usually has the largest effect of all
hyperparameters, and the best learning rate for a pretrained model can differ from the
best one for a randomly initialized model. If you tune it, tune it for each
configuration separately — and report what you used.

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
| from_scratch       | random              | —                         | VOC train / 50            | 0    | …        |
| from_coco          | COCO weights        | —                         | VOC train / 50            | 0    | …        |
| pretrained_on_coco | COCO + LightlyTrain | 25k images / 100          | VOC train / 50            | 0    | …        |

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
