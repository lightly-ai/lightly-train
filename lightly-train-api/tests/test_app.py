#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

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
    samples: list[tuple[bytes, str]],
) -> dict[str, Any]:
    response = client.post(
        "/samples",
        headers={"X-User-Id": user},
        files=[
            ("files", (f"{index}.png", data)) for index, (data, _) in enumerate(samples)
        ],
        data={"labels": [label for _, label in samples]},
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


def test_predict__no_head(client: TestClient, image_bytes: ImageBytes) -> None:
    response = client.post(
        "/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    )
    assert response.status_code == 409


def test_upload_samples__trains_and_predicts(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    body = _upload(
        client,
        "alice",
        [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4,
    )
    assert len(body["sample_ids"]) == 8

    run = client.get(f"/runs/{body['run_id']}", headers={"X-User-Id": "alice"}).json()
    assert run["status"] == "succeeded", run["error"]
    assert run["head_id"] is not None

    me = client.get("/me", headers={"X-User-Id": "alice"}).json()
    assert me["class_names"] == ["red", "blue"]
    assert me["num_samples"] == 8
    assert me["samples_per_class"] == {"red": 4, "blue": 4}
    assert me["head"]["train_accuracy"] == 1.0

    response = client.post(
        "/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    )
    prediction = response.json()[0]
    assert prediction["label"] == "red"
    assert set(prediction["probabilities"]) == {"red", "blue"}


def test_upload_samples__new_class_swaps_head(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    _upload(
        client, "alice", [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4
    )
    first = client.get("/me", headers={"X-User-Id": "alice"}).json()["head"]

    _upload(client, "alice", [(image_bytes(GREEN), "green")] * 4)
    second = client.get("/me", headers={"X-User-Id": "alice"}).json()["head"]

    assert second["id"] != first["id"]
    assert second["class_names"] == ["red", "blue", "green"]
    assert second["num_samples"] == 12

    prediction = client.post(
        "/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(GREEN)))],
    ).json()[0]
    assert prediction["label"] == "green"


def test_upload_samples__users_are_independent(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    _upload(
        client, "alice", [(image_bytes(RED), "red"), (image_bytes(BLUE), "blue")] * 4
    )
    _upload(
        client, "bob", [(image_bytes(RED), "warm"), (image_bytes(BLUE), "cold")] * 4
    )

    alice = client.post(
        "/predict",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
    ).json()[0]
    bob = client.post(
        "/predict",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
    ).json()[0]
    assert alice["label"] == "red"
    assert bob["label"] == "warm"


def test_upload_samples__label_count_mismatch(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    response = client.post(
        "/samples",
        headers={"X-User-Id": "alice"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"labels": ["red", "blue"]},
    )
    assert response.status_code == 400
