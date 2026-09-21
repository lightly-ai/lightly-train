#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from hatchet_sdk import Context, Hatchet
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from lightly_train_api import trainer
from lightly_train_api.settings import get_settings

_hatchet: Hatchet | None = None
_retrain_task: Any = None


class RetrainInput(BaseModel):
    user_id: str
    run_id: int


def get_hatchet() -> Hatchet:
    """Returns the embedded Hatchet client. The engine is per process."""
    global _hatchet
    if _hatchet is None:
        _hatchet = Hatchet.from_embedded()
    return _hatchet


def get_retrain_task() -> Any:
    global _retrain_task
    if _retrain_task is None:

        @get_hatchet().task(name="retrain", input_validator=RetrainInput)
        def retrain(input: RetrainInput, ctx: Context) -> dict[str, float]:
            return trainer.retrain_user(user_id=input.user_id, run_id=input.run_id)

        _retrain_task = retrain
    return _retrain_task


async def enqueue_retrain(user_id: str, run_id: int) -> None:
    if not get_settings().use_hatchet:
        await run_in_threadpool(trainer.retrain_user, user_id, run_id)
        return
    await get_retrain_task().aio_run_no_wait(
        RetrainInput(user_id=user_id, run_id=run_id)
    )


def create_worker() -> Any:
    """Worker for the embedded engine. Must run in the same process as the client."""
    return get_hatchet().worker("lightly-train-api", workflows=[get_retrain_task()])
