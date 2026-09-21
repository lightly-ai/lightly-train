# lightly-train-api

Draft. Serves a per-user head on top of one shared frozen pretrained model. A user is
either a classification user or a detection user, decided by their first upload.

1. `POST /samples` preprocesses the uploaded images once and stores samples plus their
   preprocessed form in SQLite.
1. A Hatchet task retrains that user's head from scratch on all their samples.
1. `POST /predict` serves the newest head.

| Task           | Model                    | Trained                              |
| -------------- | ------------------------ | ------------------------------------ |
| Classification | frozen `dinov3/vitt16`   | a linear head on the pooled features |
| Detection      | frozen `ltdetrv2-s-coco` | the class head only                  |

Datasets accumulate per user, every upload retrains the head from scratch. No
augmentations, so preprocessing happens once. The training loops live in
`lightly_train._commands.train_api`, which is an internal API.

For detection the pretrained backbone, encoder, decoder and box heads all stay frozen,
so localization stays COCO-generic and only the classes are learned. The class heads are
randomly initialized because the class count differs from the checkpoint.

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
     -F files=@dog.jpg -F labels=dog http://127.0.0.1:8000/samples
curl -H 'X-User-Id: alice' http://127.0.0.1:8000/runs/1
curl -H 'X-User-Id: alice' -F files=@cat2.jpg http://127.0.0.1:8000/predict
```

Detection, one JSON annotation per file with boxes in image pixels as `xyxy`:

```bash
curl -H 'X-User-Id: bob' \
     -F files=@street.jpg \
     -F 'annotations={"boxes": [[12, 30, 80, 140]], "labels": ["cat"]}' \
     http://127.0.0.1:8000/samples
curl -H 'X-User-Id: bob' -F files=@street2.jpg http://127.0.0.1:8000/predict
```

## Develop

```bash
make test
make check
```

Settings are environment variables with the `LIGHTLY_TRAIN_API_` prefix, see
`src/lightly_train_api/settings.py`.
