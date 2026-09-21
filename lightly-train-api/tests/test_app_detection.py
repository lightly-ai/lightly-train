#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

ImageBytes = Callable[[tuple[int, int, int]], bytes]
Annotation = Callable[[list[str]], str]

RED = (255, 0, 0)
BLUE = (0, 0, 255)


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
        data={"annotations": [item for _, item in samples]},
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_upload_samples__trains_and_predicts(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    body = _upload(
        client,
        "bob",
        [
            (image_bytes(RED), annotation(["cat"])),
            (image_bytes(BLUE), annotation(["cat", "dog"])),
        ],
    )
    assert len(body["sample_ids"]) == 2

    run = client.get(f"/runs/{body['run_id']}", headers={"X-User-Id": "bob"}).json()
    assert run["status"] == "succeeded", run["error"]
    assert run["head_id"] is not None

    me = client.get("/me", headers={"X-User-Id": "bob"}).json()
    assert me["task"] == "detection"
    assert me["class_names"] == ["cat", "dog"]
    assert me["num_samples"] == 2
    assert me["head"]["task"] == "detection"
    assert me["head"]["train_accuracy"] is None

    response = client.post(
        "/predict",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
    )
    assert response.status_code == 200, response.text
    detections = response.json()
    assert len(detections) == 1
    for box in detections[0]["boxes"]:
        assert box["label"] in {"cat", "dog"}
        assert len(box["box"]) == 4


def test_upload_samples__new_class_swaps_head(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(client, "bob", [(image_bytes(RED), annotation(["cat"]))])
    first = client.get("/me", headers={"X-User-Id": "bob"}).json()["head"]

    _upload(client, "bob", [(image_bytes(BLUE), annotation(["bird"]))])
    second = client.get("/me", headers={"X-User-Id": "bob"}).json()["head"]

    assert second["id"] != first["id"]
    assert second["class_names"] == ["cat", "bird"]
    assert second["num_samples"] == 2


def test_upload_samples__predict_boxes_are_in_image_coordinates(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(client, "bob", [(image_bytes(RED), annotation(["cat"]))])

    detections = client.post(
        "/predict",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
    ).json()

    # The uploaded images are 64x64, boxes are denormalized back to that size.
    for box in detections[0]["boxes"]:
        assert all(-64.0 <= value <= 128.0 for value in box["box"])


def test_upload_samples__annotation_count_mismatch(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    response = client.post(
        "/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"annotations": [annotation(["cat"]), annotation(["dog"])]},
    )

    assert response.status_code == 400


def test_upload_samples__invalid_annotation(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    response = client.post(
        "/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"annotations": [json.dumps({"boxes": [[1, 2, 3]], "labels": ["cat"]})]},
    )

    assert response.status_code == 400


def test_upload_samples__labels_and_annotations(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    response = client.post(
        "/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"labels": ["cat"], "annotations": [annotation(["cat"])]},
    )

    assert response.status_code == 400


def test_upload_samples__task_cannot_change(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(client, "bob", [(image_bytes(RED), annotation(["cat"]))])

    response = client.post(
        "/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(BLUE)))],
        data={"labels": ["cat"]},
    )

    assert response.status_code == 400
