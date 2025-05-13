#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from datetime import datetime, timezone

from lightly_train._checkpoint import (
    CheckpointLightlyTrain,
    CheckpointLightlyTrainModels,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs

from . import helpers


class TestCheckpointInfo:
    def test_to_dict_from_dict(self) -> None:
        date = datetime(
            year=2024, month=1, day=2, minute=3, second=4, tzinfo=timezone.utc
        )
        wrapped_model = helpers.DummyCustomModel()
        embedding_model = EmbeddingModel(wrapped_model=wrapped_model)
        info = CheckpointLightlyTrain(
            version="abc",
            date=date,
            models=CheckpointLightlyTrainModels(
                model=wrapped_model.get_model(),
                wrapped_model=wrapped_model,
                embedding_model=embedding_model,
            ),
            normalize_args=NormalizeArgs(),
        )

        # Check that to_dict representation is correct.
        info_dict = info.to_dict()
        assert info_dict == {
            "version": "abc",
            "date": "2024-01-02T00:03:04+00:00",
            "models": {
                "model": wrapped_model.get_model(),
                "wrapped_model": wrapped_model,
                "embedding_model": embedding_model,
            },
            "normalize_args": NormalizeArgs().to_dict(),
        }

        # Check that from_dict reconstruction is correct.
        assert info == CheckpointLightlyTrain.from_dict(info_dict)
