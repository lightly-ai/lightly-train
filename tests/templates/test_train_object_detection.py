#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

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
        assert 'batch_size="32"' in result
        assert 'num_workers="4"' in result
        assert 'model="rtdetr_r18vd_6x_coco"' in result
        assert "steps=1000" in result
        assert 'precision="32"' in result
        assert "seed=42" in result
        assert 'devices="2"' in result
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
        assert "steps=None" in result
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
