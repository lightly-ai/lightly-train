#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from hatchet_sdk import Context, Hatchet, Worker
from hatchet_sdk.runnables.workflow import Standalone
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from lightly_train_api import trainer
from lightly_train_api.settings import get_settings


class RetrainInput(BaseModel):
    dataset_id: int
    run_id: int


RetrainTask = Standalone[RetrainInput, trainer.TrainMetrics]

_hatchet: Hatchet | None = None
_retrain_task: RetrainTask | None = None


def get_hatchet() -> Hatchet:
    """Returns the embedded Hatchet client. The engine is per process."""
    global _hatchet
    if _hatchet is None:
        _hatchet = Hatchet.from_embedded()
    return _hatchet


def get_retrain_task() -> RetrainTask:
    global _retrain_task
    if _retrain_task is None:

        @get_hatchet().task(name="retrain", input_validator=RetrainInput)
        def retrain(input: RetrainInput, ctx: Context) -> trainer.TrainMetrics:
            return trainer.retrain_dataset(
                dataset_id=input.dataset_id, run_id=input.run_id
            )

        _retrain_task = retrain
    return _retrain_task


async def enqueue_retrain(dataset_id: int, run_id: int) -> None:
    if not get_settings().use_hatchet:
        await run_in_threadpool(trainer.retrain_dataset, dataset_id, run_id)
        return
    await get_retrain_task().aio_run_no_wait(
        RetrainInput(dataset_id=dataset_id, run_id=run_id)
    )


def create_worker() -> Worker:
    """Worker for the embedded engine. Must run in the same process as the client."""
    return get_hatchet().worker("lightly-train-api", workflows=[get_retrain_task()])
