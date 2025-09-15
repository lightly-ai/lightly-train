from lightly_train._transforms.random_order import RandomOrder
from albumentations import Resize, ColorJitter
import pytest
import numpy as np

class TestRandomOrder:
    def test__shapes(self) -> None:
        tr = RandomOrder(
            transforms=[Resize(32, 32, p=1.0), ColorJitter(brightness=0.5, contrast=0.5)],
            n=2,
            p=1.0,
        )
        img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        out = tr(image=img)
        assert out["image"].shape == (32, 32, 3)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test__get_idx(self, n: int) -> None:
        tr = RandomOrder(
            transforms=[Resize(32, 32, p=1.0), ColorJitter(brightness=0.5, contrast=0.5)],
            n=n,
            p=1.0,
        )
        if n == 0:
            assert set(tr._get_idx().tolist()) == set()
        elif n == 1:
            assert set(tr._get_idx().tolist()).issubset({0, 1})
        elif n == 2:
            assert set(tr._get_idx().tolist()) == {0, 1}