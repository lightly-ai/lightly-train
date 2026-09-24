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
    dataset: str,
    samples: list[tuple[bytes, str]],
    keys: list[str] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"annotations": [item for _, item in samples]}
    if keys is not None:
        data["keys"] = keys
    response = client.post(
        f"/datasets/{dataset}/samples",
        headers={"X-User-Id": user},
        files=[
            ("files", (keys[index] if keys else f"{index}.png", image))
            for index, (image, _) in enumerate(samples)
        ],
        data=data,
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
        "street",
        [
            (image_bytes(RED), annotation(["cat"])),
            (image_bytes(BLUE), annotation(["cat", "dog"])),
        ],
    )
    assert len(body["ingested"]) == 2

    run = client.get(f"/runs/{body['run_id']}", headers={"X-User-Id": "bob"}).json()
    assert run["status"] == "succeeded", run["error"]
    assert run["head_id"] is not None

    info = client.get("/datasets/street", headers={"X-User-Id": "bob"}).json()
    assert info["task"] == "detection"
    assert info["class_names"] == ["cat", "dog"]
    assert info["num_samples"] == 2
    assert info["head"]["task"] == "detection"
    assert info["head"]["train_accuracy"] is None

    response = client.post(
        "/datasets/street/predict",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
    )
    assert response.status_code == 200, response.text
    detections = response.json()
    assert len(detections) == 1
    for box in detections[0]["boxes"]:
        assert box["label"] in {"cat", "dog"}
        assert len(box["box"]) == 4


def test_upload_samples__reupload_is_skipped(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    samples = [(image_bytes(RED), annotation(["cat"]))]
    _upload(client, "bob", "street", samples, keys=["a.png"])

    body = _upload(client, "bob", "street", samples, keys=["a.png"])

    assert body["unchanged"] == ["a.png"]
    assert body["run_id"] is None


def test_upload_samples__changed_boxes_update_sample(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    image = image_bytes(RED)
    _upload(client, "bob", "street", [(image, annotation(["cat"]))], keys=["a.png"])

    body = _upload(
        client, "bob", "street", [(image, annotation(["cat", "dog"]))], keys=["a.png"]
    )

    assert body["updated"] == ["a.png"]
    assert body["num_samples"] == 1
    assert body["class_names"] == ["cat", "dog"]


def test_upload_samples__new_class_swaps_head(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(
        client,
        "bob",
        "street",
        [(image_bytes(RED), annotation(["cat"]))],
        keys=["a.png"],
    )
    first = client.get("/datasets/street", headers={"X-User-Id": "bob"}).json()["head"]

    _upload(
        client,
        "bob",
        "street",
        [(image_bytes(BLUE), annotation(["bird"]))],
        keys=["b.png"],
    )
    second = client.get("/datasets/street", headers={"X-User-Id": "bob"}).json()["head"]

    assert second["id"] != first["id"]
    assert second["class_names"] == ["cat", "bird"]
    assert second["num_samples"] == 2


def test_upload_samples__predict_boxes_are_in_image_coordinates(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(client, "bob", "street", [(image_bytes(RED), annotation(["cat"]))])

    detections = client.post(
        "/datasets/street/predict",
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
        "/datasets/street/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"annotations": [annotation(["cat"]), annotation(["dog"])]},
    )

    assert response.status_code == 400


def test_upload_samples__invalid_annotation(
    client: TestClient, image_bytes: ImageBytes
) -> None:
    response = client.post(
        "/datasets/street/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"annotations": [json.dumps({"boxes": [[1, 2, 3]], "labels": ["cat"]})]},
    )

    assert response.status_code == 400


def test_upload_samples__labels_and_annotations(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    response = client.post(
        "/datasets/street/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("a.png", image_bytes(RED)))],
        data={"labels": ["cat"], "annotations": [annotation(["cat"])]},
    )

    assert response.status_code == 400


def test_upload_samples__task_cannot_change(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(client, "bob", "street", [(image_bytes(RED), annotation(["cat"]))])

    response = client.post(
        "/datasets/street/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("b.png", image_bytes(BLUE)))],
        data={"labels": ["cat"]},
    )

    assert response.status_code == 400


def test_upload_samples__task_can_differ_per_dataset(
    client: TestClient, image_bytes: ImageBytes, annotation: Annotation
) -> None:
    _upload(client, "bob", "street", [(image_bytes(RED), annotation(["cat"]))])

    response = client.post(
        "/datasets/colors/samples",
        headers={"X-User-Id": "bob"},
        files=[("files", ("b.png", image_bytes(BLUE)))],
        data={"labels": ["blue"]},
    )

    assert response.status_code == 200, response.text
    assert (
        client.get("/datasets/colors", headers={"X-User-Id": "bob"}).json()["task"]
        == "classification"
    )
