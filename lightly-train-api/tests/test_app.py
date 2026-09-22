#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

ImageBytes = Callable[[tuple[int, int, int]], bytes]

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)


def _upload(
    client: TestClient,
    user: str,
    dataset: str,
    samples: list[tuple[bytes, str]],
    keys: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"labels": [label for _, label in samples]}
    if keys is not None:
        data["keys"] = keys
    response = client.post(
        f"/datasets/{dataset}/samples",
        headers={"X-User-Id": user},
        files=[
            ("files", (keys[index] if keys else f"{index}.png", data_))
            for index, (data_, _) in enumerate(samples)
        ],
        data=data,
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client: TestClient) -> None:
    body = client.get("/model").json()
    assert body["backbone"] == "dinov3/vitt16"
    assert body["feature_dim"] == 192


def test_get_dataset__unknown(client: TestClient) -> None:
    response = client.get("/datasets/missing", headers={"X-User-Id": "alice"})
    assert response.status_code == 404


def test_predict__no_dataset(client: TestClient, image_bytes: ImageBytes) -> None:
    response = client.post(
        "/datasets/missing/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    )
    assert response.status_code == 404


def test_upload_samples__trains_and_predicts(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    body = _upload(
        client,
        "alice",
        "colors",
        [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4,
    )
    assert len(body["ingested"]) == 8
    assert body["unchanged"] == []
    assert body["num_samples"] == 8

    run = client.get(f"/runs/{body['run_id']}", headers={"X-User-Id": "alice"}).json()
    assert run["status"] == "succeeded", run["error"]
    assert run["head_id"] is not None
    assert run["dataset"] == "colors"

    info = client.get("/datasets/colors", headers={"X-User-Id": "alice"}).json()
    assert info["class_names"] == ["red", "blue"]
    assert info["num_samples"] == 8
    assert info["samples_per_class"] == {"red": 4, "blue": 4}
    assert info["head"]["train_accuracy"] == 1.0

    response = client.post(
        "/datasets/colors/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    )
    prediction = response.json()[0]
    assert prediction["label"] == "red"
    assert set(prediction["probabilities"]) == {"red", "blue"}


def test_upload_samples__new_class_swaps_head(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    samples = [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4
    _upload(client, "alice", "colors", samples, keys=[f"a{i}.png" for i in range(8)])
    first = client.get("/datasets/colors", headers={"X-User-Id": "alice"}).json()[
        "head"
    ]

    _upload(
        client,
        "alice",
        "colors",
        [(image_bytes(GREEN), "green")] * 4,
        keys=[f"b{i}.png" for i in range(4)],
    )
    second = client.get("/datasets/colors", headers={"X-User-Id": "alice"}).json()[
        "head"
    ]

    assert second["id"] != first["id"]
    assert second["class_names"] == ["red", "blue", "green"]
    assert second["num_samples"] == 12

    prediction = client.post(
        "/datasets/colors/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(GREEN)))],
    ).json()[0]
    assert prediction["label"] == "green"


def test_upload_samples__reupload_is_skipped(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    samples = [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4
    keys = [f"img_{index}.png" for index in range(len(samples))]
    first = _upload(client, "alice", "colors", samples, keys=keys)
    assert len(first["ingested"]) == 8

    second = _upload(client, "alice", "colors", samples, keys=keys)

    assert second["ingested"] == []
    assert second["updated"] == []
    assert len(second["unchanged"]) == 8
    assert second["num_samples"] == 8
    # Nothing changed, so no training run is started.
    assert second["run_id"] is None


def test_upload_samples__changed_label_updates_sample(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    samples = [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4
    keys = [f"img_{index}.png" for index in range(len(samples))]
    _upload(client, "alice", "colors", samples, keys=keys)

    relabeled = [
        (data, "crimson" if label == "red" else label) for data, label in samples
    ]
    body = _upload(client, "alice", "colors", relabeled, keys=keys)

    assert len(body["updated"]) == 4
    assert body["ingested"] == []
    # Still eight samples, the four red ones were updated in place.
    assert body["num_samples"] == 8
    assert body["class_names"] == ["red", "blue", "crimson"]
    assert body["run_id"] is not None


def test_upload_samples__changed_image_updates_sample(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    keys = ["a.png", "b.png"]
    _upload(
        client,
        "alice",
        "colors",
        [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")],
        keys=keys,
    )

    body = _upload(
        client,
        "alice",
        "colors",
        [(image_bytes(GREEN), "red"), (image_bytes(BLUE), "blue")],
        keys=keys,
    )

    assert body["updated"] == ["a.png"]
    assert body["unchanged"] == ["b.png"]
    assert body["num_samples"] == 2


def test_diff_samples(client: TestClient, image_bytes: ImageBytes) -> None:
    red, blue = image_bytes(RED), image_bytes(BLUE)
    _upload(
        client,
        "alice",
        "colors",
        [(red, "red"), (blue, "blue")],
        keys=["a.png", "b.png"],
    )

    response = client.post(
        "/datasets/colors/samples/diff",
        headers={"X-User-Id": "alice"},
        json={
            "samples": [
                {
                    "key": "a.png",
                    "content_hash": hashlib.sha256(red).hexdigest(),
                    "label": "red",
                },
                {
                    "key": "b.png",
                    "content_hash": hashlib.sha256(blue).hexdigest()[::-1],
                    "label": "blue",
                },
                {"key": "c.png", "content_hash": "whatever", "label": "green"},
            ]
        },
    )

    assert response.json() == {
        "new": ["c.png"],
        "changed": ["b.png"],
        "unchanged": ["a.png"],
    }


def test_diff_samples__relabeled_image_is_changed(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    red = image_bytes(RED)
    _upload(client, "alice", "colors", [(red, "red")], keys=["a.png"])

    response = client.post(
        "/datasets/colors/samples/diff",
        headers={"X-User-Id": "alice"},
        json={
            "samples": [
                {
                    "key": "a.png",
                    "content_hash": hashlib.sha256(red).hexdigest(),
                    "label": "crimson",
                }
            ]
        },
    )

    assert response.json()["changed"] == ["a.png"]


def test_diff_samples__unknown_dataset(client: TestClient) -> None:
    response = client.post(
        "/datasets/missing/samples/diff",
        headers={"X-User-Id": "alice"},
        json={"samples": [{"key": "a.png", "content_hash": "x", "label": "red"}]},
    )

    assert response.json()["new"] == ["a.png"]


def test_upload_samples__datasets_are_independent(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    _upload(
        client,
        "alice",
        "colors",
        [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4,
    )
    _upload(
        client,
        "alice",
        "temperature",
        [(image_bytes(RED), "warm"), (image_bytes(BLUE), "cold")] * 4,
    )

    assert client.get("/datasets", headers={"X-User-Id": "alice"}).json() == [
        "colors",
        "temperature",
    ]
    colors = client.post(
        "/datasets/colors/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    ).json()[0]
    temperature = client.post(
        "/datasets/temperature/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    ).json()[0]
    assert colors["label"] == "red"
    assert temperature["label"] == "warm"


def test_upload_samples__users_are_independent(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    _upload(
        client,
        "alice",
        "colors",
        [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4,
    )
    _upload(
        client,
        "bob",
        "colors",
        [(image_bytes(RED), "warm"), (image_bytes(BLUE), "cold")] * 4,
    )

    assert client.get("/datasets", headers={"X-User-Id": "bob"}).json() == ["colors"]
    bob = client.post(
        "/datasets/colors/predict",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
    ).json()[0]
    assert bob["label"] == "warm"


def test_get_run__other_user(client: TestClient, image_bytes: ImageBytes) -> None:
    body = _upload(client, "alice", "colors", [(image_bytes(RED), "red")])

    response = client.get(f"/runs/{body['run_id']}", headers={"X-User-Id": "bob"})

    assert response.status_code == 404


def test_upload_samples__label_count_mismatch(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    response = client.post(
        "/datasets/colors/samples",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"labels": ["red", "blue"]},
    )
    assert response.status_code == 400


def test_upload_samples__duplicate_keys(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    response = client.post(
        "/datasets/colors/samples",
        headers={"X-User-Id": "alice"},
        files=[
            ("files", ("a.png", image_bytes(RED))),
            ("files", ("a.png", image_bytes(BLUE))),
        ],
        data={"labels": ["red", "blue"], "keys": ["same", "same"]},
    )
    assert response.status_code == 400


def test_upload_samples__key_count_mismatch(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    response = client.post(
        "/datasets/colors/samples",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"labels": ["red"], "keys": ["a", "b"]},
    )
    assert response.status_code == 400
