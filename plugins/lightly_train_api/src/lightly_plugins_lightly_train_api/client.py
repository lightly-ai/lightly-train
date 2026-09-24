"""HTTP client for the LightlyTrain API service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Tuple

import httpx

# Terminal states of a training run, as reported by `GET /runs/{id}`.
FINISHED = "succeeded"
FAILED = "failed"
TERMINAL = frozenset({FINISHED, FAILED})

# Multipart payload of one image: (field name, (filename, bytes)).
FilePart = Tuple[str, Tuple[str, bytes]]


class ApiError(RuntimeError):
    """Raised when the API answers with an error status."""


@dataclass(frozen=True)
class ApiClient:
    """Talks to one LightlyTrain API service as one user."""

    url: str
    user_id: str
    timeout: float = 120.0

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.url.rstrip('/')}{path}",
                headers={"X-User-Id": self.user_id},
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.HTTPError as error:
            raise ApiError(f"Cannot reach {self.url}: {error}") from error
        if response.status_code >= 400:
            raise ApiError(
                f"{method} {path} failed with {response.status_code}: {response.text}"
            )
        return response.json()

    def get_dataset(self, dataset: str) -> dict[str, Any] | None:
        """Returns the dataset, or None if the server does not know it yet."""
        try:
            info: dict[str, Any] = self._request("GET", f"/datasets/{dataset}")
        except ApiError as error:
            if " 404:" in str(error):
                return None
            raise
        return info

    def diff(self, dataset: str, samples: list[dict[str, Any]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = self._request(
            "POST", f"/datasets/{dataset}/samples/diff", json={"samples": samples}
        )
        return result

    def upload(
        self,
        dataset: str,
        files: list[FilePart],
        keys: list[str],
        labels: list[str] | None = None,
        annotations: list[str] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"keys": keys}
        if labels is not None:
            data["labels"] = labels
        if annotations is not None:
            data["annotations"] = annotations
        result: dict[str, Any] = self._request(
            "POST", f"/datasets/{dataset}/samples", files=files, data=data
        )
        return result

    def predict(
        self, dataset: str, files: list[FilePart], threshold: float
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "POST",
            f"/datasets/{dataset}/predict",
            files=files,
            data={"threshold": threshold},
        )
        return result

    def get_run(self, run_id: int) -> dict[str, Any]:
        run: dict[str, Any] = self._request("GET", f"/runs/{run_id}")
        return run

    def wait_for_run(
        self, run_id: int, timeout_s: float, poll_s: float = 2.0
    ) -> dict[str, Any]:
        """Polls a run until it reaches a terminal state or the timeout expires."""
        deadline = time.monotonic() + timeout_s
        run = self.get_run(run_id)
        while run["status"] not in TERMINAL:
            if time.monotonic() > deadline:
                return run
            time.sleep(poll_s)
            run = self.get_run(run_id)
        return run
