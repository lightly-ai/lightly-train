#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from .. import helpers

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


def _render(**kwargs: object) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("train_object_detection.jinja2")
    return template.render(**kwargs)


class TestTrainObjectDetectionTemplate:
    def test_render_with_required_params_only(self) -> None:
        result = _render(
            out="/output",
            train_annotations="train.json",
            val_annotations="val.json",
        )
        assert 'out="/output"' in result
        assert '"annotations": "train.json"' in result
        assert '"annotations": "val.json"' in result
        # Images should default to None when not provided.
        assert '"images": None' in result

    def test_render_with_all_params(self) -> None:
        result = _render(
            out="/output",
            overwrite=True,
            train_annotations="train.json",
            train_images="/train_imgs",
            val_annotations="val.json",
            val_images="/val_imgs",
            skip_if_annotations_missing=False,
            batch_size=32,
            num_workers=4,
            model="rtdetr_r18vd_6x_coco",
            steps=1000,
            precision="32",
            seed=42,
            devices=2,
            accelerator="gpu",
            num_nodes=2,
            strategy="ddp",
            resume_interrupted=True,
            model_args={"lr": 0.001},
            save_checkpoint_args={"save_last": True},
            logger_args={"tensorboard": {}},
            transform_args={"image_size": 640},
            metric_args={"classwise": True},
            torch_compile_args={"mode": "default"},
        )
        assert 'out="/output"' in result
        assert "overwrite=True" in result
        assert '"images": "/train_imgs"' in result
        assert '"images": "/val_imgs"' in result
        assert '"skip_if_annotations_missing": False' in result
        assert "batch_size=32" in result
        assert "num_workers=4" in result
        assert 'model="rtdetr_r18vd_6x_coco"' in result
        assert "steps=1000" in result
        assert 'precision="32"' in result
        assert "seed=42" in result
        assert "devices=2" in result
        assert 'accelerator="gpu"' in result
        assert "num_nodes=2" in result
        assert 'strategy="ddp"' in result
        assert "resume_interrupted=True" in result
        assert "model_args={'lr': 0.001}" in result
        assert "save_checkpoint_args={'save_last': True}" in result
        assert "logger_args={'tensorboard': {}}" in result
        assert "transform_args={'image_size': 640}" in result
        assert "metric_args={'classwise': True}" in result
        assert "torch_compile_args={'mode': 'default'}" in result

    def test_default_values(self) -> None:
        result = _render(
            out="/output",
            train_annotations="train.json",
            val_annotations="val.json",
        )
        assert "overwrite=False" in result
        assert '"skip_if_annotations_missing": True' in result
        assert 'batch_size="auto"' in result
        assert 'num_workers="auto"' in result
        assert 'model="dinov3/vitt16-ltdetr-coco"' in result
        assert "model_args=None" in result
        assert 'steps="auto"' in result
        assert 'precision="bf16-mixed"' in result
        assert "seed=0" in result
        assert 'devices="auto"' in result
        assert 'accelerator="auto"' in result
        assert "num_nodes=1" in result
        assert 'strategy="auto"' in result
        assert "resume_interrupted=False" in result
        assert "save_checkpoint_args=None" in result
        assert "logger_args=None" in result
        assert "transform_args=None" in result
        assert "metric_args=None" in result
        assert "torch_compile_args=None" in result

    def test_seed_zero_is_preserved(self) -> None:
        """Passing seed=0 explicitly should render 0, not the default."""
        result = _render(
            out="/output",
            train_annotations="train.json",
            val_annotations="val.json",
            seed=0,
        )
        assert "seed=0" in result

    def test_overwrite_false_is_preserved(self) -> None:
        """Passing overwrite=False explicitly should render False, not the default."""
        result = _render(
            out="/output",
            train_annotations="train.json",
            val_annotations="val.json",
            overwrite=False,
        )
        assert "overwrite=False" in result

    def test_images_none_when_not_provided(self) -> None:
        result = _render(
            out="/output",
            train_annotations="train.json",
            val_annotations="val.json",
        )
        # Should render None without quotes, not "None".
        assert '"images": None' in result
        assert '"images": "None"' not in result

    def test_rendered_output_is_valid_python_syntax(self) -> None:
        result = _render(
            out="/output",
            train_annotations="train.json",
            val_annotations="val.json",
        )
        # Should parse without SyntaxError.
        compile(result, "<template>", "exec")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Slow on windows")
def test_rendered_template_runs_training_with_defaults(tmp_path: Path) -> None:
    """Integration test: render with default parameters (except model) and run."""
    data = tmp_path / "data"
    out = tmp_path / "out"
    helpers.create_coco_object_detection_dataset(data, num_files=4)

    template_args = dict(
        out=str(out),
        train_annotations=str(data / "train.json"),
        train_images=str(data / "train"),
        val_annotations=str(data / "val.json"),
        val_images=str(data / "val"),
        # Override model, steps, batch_size, num_workers, and devices to keep the test fast.
        model="dinov3/vitt16-notpretrained-ltdetr",
        steps=2,
        batch_size=2,
        num_workers=2,
        devices=1,
    )
    if sys.platform.startswith("darwin"):
        template_args["accelerator"] = "cpu"

    result = _render(**template_args)

    exec(compile(result, "<template>", "exec"))

    assert out.exists()
    assert (out / "train.log").exists()
    assert (out / "exported_models" / "exported_last.pt").exists()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Slow on windows")
def test_rendered_template_runs_training_with_all_params(tmp_path: Path) -> None:
    """Integration test: render with all parameters set explicitly and run."""
    data = tmp_path / "data"
    out = tmp_path / "out"
    helpers.create_coco_object_detection_dataset(data, num_files=4)

    result = _render(
        out=str(out),
        overwrite=False,
        train_annotations=str(data / "train.json"),
        train_images=str(data / "train"),
        val_annotations=str(data / "val.json"),
        val_images=str(data / "val"),
        skip_if_annotations_missing=True,
        batch_size=2,
        num_workers=2,
        model="dinov3/vitt16-notpretrained-ltdetr",
        model_args=None,
        steps=2,
        precision="32",
        seed=42,
        devices=1,
        accelerator="cpu",
        num_nodes=1,
        strategy="auto",
        resume_interrupted=False,
        save_checkpoint_args={"save_last": True},
        logger_args={"log_every_num_steps": 1},
        transform_args={"image_size": "auto"},
        metric_args={"classwise": False},
        torch_compile_args={"disable": True},
    )

    exec(compile(result, "<template>", "exec"))

    assert out.exists()
    assert (out / "train.log").exists()
    assert (out / "exported_models" / "exported_last.pt").exists()
