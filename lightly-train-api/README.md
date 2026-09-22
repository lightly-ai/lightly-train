# lightly-train-api

Draft. Serves a per-dataset head on top of one shared frozen pretrained model. Samples
arrive over the API with a user and a dataset name, and a complete roundtrip is:

1. `POST /datasets/{name}/samples/diff` reports which samples the server is missing.
1. `POST /datasets/{name}/samples` ingests those, preprocesses them once and stores them
   in SQLite. Samples already known are skipped.
1. A Hatchet task retrains that dataset's head from scratch on all of its samples.
1. `POST /datasets/{name}/predict` serves the newest head.

Each `(user, dataset)` pair owns its own samples, class names and head. A dataset is
either a classification dataset or a detection dataset, decided by its first upload.

| Task           | Model                    | Trained                              |
| -------------- | ------------------------ | ------------------------------------ |
| Classification | frozen `dinov3/vitt16`   | a linear head on the pooled features |
| Detection      | frozen `ltdetrv2-s-coco` | the class head only                  |

Datasets accumulate, every ingest that changes something retrains the head from scratch.
No augmentations, so preprocessing happens once. The training loops live in
`lightly_train._commands.train_api`, which is an internal API.

For detection the pretrained backbone, encoder, decoder and box heads all stay frozen,
so localization stays COCO-generic and only the classes are learned. The class heads are
randomly initialized because the class count differs from the checkpoint.

## Samples

A sample is identified by its `key` within a dataset, which defaults to the uploaded
filename. Re-uploading the same key replaces the sample instead of adding a second one,
so a client can push its whole dataset on every run:

- unknown key: ingested
- known key, same image and same annotation: skipped
- known key, different image or annotation: updated, and the head is retrained

`/samples/diff` applies the same comparison without the image bytes, so a client can ask
what to upload before sending anything.

## Run

```bash
uv sync
uv run python -m lightly_train_api   # http://127.0.0.1:8000/docs
```

Hatchet runs embedded: the engine and its Postgres are started as a sidecar on first
use, and the worker runs inside the API process. Set `LIGHTLY_TRAIN_API_USE_HATCHET=0`
to retrain inline instead.

Classification, one label per file:

```bash
curl -H 'X-User-Id: alice' -F files=@cat.jpg -F labels=cat \
     -F files=@dog.jpg -F labels=dog http://127.0.0.1:8000/datasets/pets/samples
curl -H 'X-User-Id: alice' http://127.0.0.1:8000/runs/1
curl -H 'X-User-Id: alice' http://127.0.0.1:8000/datasets/pets
curl -H 'X-User-Id: alice' -F files=@cat2.jpg \
     http://127.0.0.1:8000/datasets/pets/predict
```

Detection, one JSON annotation per file with boxes in image pixels as `xyxy`:

```bash
curl -H 'X-User-Id: bob' \
     -F files=@street.jpg \
     -F 'annotations={"boxes": [[12, 30, 80, 140]], "labels": ["cat"]}' \
     http://127.0.0.1:8000/datasets/street/samples
curl -H 'X-User-Id: bob' -F files=@street2.jpg \
     http://127.0.0.1:8000/datasets/street/predict
```

## LightlyStudio

Two LightlyStudio operators drive this service, see
[plugins/lightly_train_api](../plugins/lightly_train_api/): one uploads the annotated
images of a view and trains, the other predicts a view and writes the predictions back
as annotations.

## Develop

```bash
make test
make check
```

Settings are environment variables with the `LIGHTLY_TRAIN_API_` prefix, see
`src/lightly_train_api/settings.py`.
