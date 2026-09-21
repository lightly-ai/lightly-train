# lightly-train-api

Draft. Serves a per-user linear classifier on top of one shared frozen `dinov3/vitt16`
backbone.

1. `POST /samples` encodes the uploaded images once and stores samples plus embeddings
   in SQLite.
1. A Hatchet task retrains that user's linear head from scratch on all their embeddings.
1. `POST /predict` serves the newest head.

Datasets accumulate per user, every upload retrains the head from scratch. No
augmentations, so embeddings are computed once. Built on `lightly_train`'s
`ImageClassification`, which is an internal API.

## Run

```bash
uv sync
uv run python -m lightly_train_api   # http://127.0.0.1:8000/docs
```

Hatchet runs embedded: the engine and its Postgres are started as a sidecar on first
use, and the worker runs inside the API process. Set `LIGHTLY_TRAIN_API_USE_HATCHET=0`
to retrain inline instead.

```bash
curl -H 'X-User-Id: alice' -F files=@cat.jpg -F labels=cat \
     -F files=@dog.jpg -F labels=dog http://127.0.0.1:8000/samples
curl -H 'X-User-Id: alice' http://127.0.0.1:8000/runs/1
curl -H 'X-User-Id: alice' -F files=@cat2.jpg http://127.0.0.1:8000/predict
```

## Develop

```bash
make test
make check
```

Settings are environment variables with the `LIGHTLY_TRAIN_API_` prefix, see
`src/lightly_train_api/settings.py`.
