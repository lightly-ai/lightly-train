#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass
from json import JSONEncoder
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    Literal,
    Mapping,
    Type,
    TypeVar,
)

import torch
from filelock import FileLock
from lightning_fabric import Fabric
from lightning_fabric import utilities as fabric_utilities
from lightning_fabric.loggers.logger import Logger as FabricLogger
from torch import Tensor
from torch.utils.data import DataLoader

from lightly_train._configs import validate
from lightly_train._data import cache
from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import (
    MemoryMappedSequence,
    Primitive,
)
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train._data.task_dataset import TaskDataset
from lightly_train._env import Env
from lightly_train._loggers.mlflow import MLFlowLogger, MLFlowLoggerArgs
from lightly_train._loggers.task_logger_args import TaskLoggerArgs
from lightly_train._loggers.tensorboard import TensorBoardLogger
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.dinov2_eomt_semantic_segmentation.train_model import (
    DINOv2EoMTSemanticSegmentationTrain,
)
from lightly_train._task_models.dinov2_linear_semantic_segmentation.train_model import (
    DINOv2LinearSemanticSegmentationTrain,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import (
    DINOv3EoMTSemanticSegmentationTrain,
)
from lightly_train._task_models.train_model import (
    TrainModel,
    TrainModelArgs,
)
from lightly_train._train_task_state import TrainTaskState
from lightly_train._transforms.semantic_segmentation_transform import (
    SemanticSegmentationTransform,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
)
from lightly_train.types import (
    PathLike,
    TaskDatasetItem,
)

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lightly_train._commands.train_task import TrainTaskConfig


_MergeT = TypeVar(
    "_MergeT",
    TrainModelArgs,
    MaskSemanticSegmentationDataArgs,
    TaskTransformArgs,
)


def _merge(
    model_cls: Type[_MergeT],
    current_obj: _MergeT,
    update_source: dict[str, Any],
) -> _MergeT:
    """Merge checkpoint values into a Pydantic model and revalidate."""
    merged_dict = current_obj.model_dump()
    for key in model_cls.model_fields:
        if key in update_source:
            merged_dict[key] = update_source[key]
    return validate.pydantic_model_validate(model_cls, merged_dict)


@dataclass
class MetadataMergeResult:
    model_args: TrainModelArgs
    data_args: MaskSemanticSegmentationDataArgs
    train_transform_args: TaskTransformArgs
    val_transform_args: TaskTransformArgs


@dataclass
class CheckpointContext:
    path: Path
    mode: Literal["finetune", "resume"]
    metadata: dict[str, Any]
    train_model_state: dict[str, Tensor] | None
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    step: int | None = None

    @classmethod
    def from_config(
        cls, *, fabric: Fabric, config: "TrainTaskConfig", out_dir: Path
    ) -> "CheckpointContext | None":
        """Build a checkpoint context from the current run configuration.

        Args:
            fabric: Fabric instance used to load checkpoint files.
            config: Training configuration holding resume and checkpoint flags.
            out_dir: Output directory where checkpoints are stored.

        Returns:
            A populated checkpoint context when resuming or fine-tuning, otherwise ``None``.

        Raises:
            ValueError: If resume and checkpoint options are requested simultaneously.
            FileNotFoundError: If the resolved checkpoint file does not exist.
        """
        if (not config.resume_interrupted) and (config.checkpoint is None):
            # Not resuming or fine-tuning from a checkpoint.
            return None
        elif config.resume_interrupted and (config.checkpoint is not None):
            # Both resume and checkpoint specified.
            raise ValueError(
                f"resume_interrupted={config.resume_interrupted} and checkpoint='{config.checkpoint}' "
                "cannot be set at the same time! Please set only one of them. "
            )
        elif config.checkpoint is not None:
            ckpt_path = Path(config.checkpoint).resolve()
        else:
            ckpt_path = get_checkpoint_path(out_dir, best_or_last="last")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' does not exist.")

        logger.info(f"Loading metadata of the checkpoint from '{ckpt_path}'")

        checkpoint = fabric.load(path=ckpt_path)
        mode: Literal["finetune", "resume"] = (
            "resume" if config.resume_interrupted else "finetune"
        )
        metadata = {
            "model_init_args": checkpoint.get("model_init_args") or {},
            "model_class_path": checkpoint.get("model_class_path") or "",
        }
        return cls(
            path=ckpt_path,
            mode=mode,
            metadata=metadata,
            train_model_state=checkpoint.get("train_model"),
            optimizer_state=checkpoint.get("optimizer"),
            scheduler_state=checkpoint.get("scheduler"),
            step=checkpoint.get("step"),
        )

    def apply_metadata(
        self,
        *,
        model_args: TrainModelArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        train_transform_args: TaskTransformArgs,
        val_transform_args: TaskTransformArgs,
        config_model: str,
        config_steps: int | Literal["auto"],
    ) -> MetadataMergeResult:
        """Merge checkpoint metadata into the current task configuration.

        Args:
            model_args: Pydantic model arguments for the train model.
            data_args: Semantic segmentation data arguments.
            train_transform_args: Training transform arguments.
            val_transform_args: Validation transform arguments.
            config_model: Model name defined in the active configuration.
            config_steps: Target number of optimization steps.

        Returns:
            Updated configuration values derived from the checkpoint metadata.

        Raises:
            ValueError: If the checkpoint references a different model or incompatible
                settings in resume mode.
        """
        model_init_args: dict[str, Any] = self.metadata.get("model_init_args", {})
        if not model_init_args:
            # No metadata to apply from checkpoint.
            return MetadataMergeResult(
                model_args=model_args,
                data_args=data_args,
                train_transform_args=train_transform_args,
                val_transform_args=val_transform_args,
            )

        checkpoint_model_name = model_init_args.get("model_name")
        if checkpoint_model_name and checkpoint_model_name != config_model:
            raise ValueError(
                "The checkpoint was created with model_name="
                f"'{checkpoint_model_name}', but config.model='{config_model}'. "
                "Please use a checkpoint that matches the configured model or "
                "update config.model to the correct value before resuming."
            )

        model_args_dict = model_args.model_dump()
        data_args_dict = data_args.model_dump()
        # BUG: the train model is using image_size and image_normalize from the val_transform_args here instead of train_transform_args
        # which will cause issues if we specify different values for train and val transforms in transform_args input.
        # So for now we only check if there is a mismatch in val_transform_args
        # train_transform_args_dict = train_transform_args.model_dump()
        val_transform_args_dict = val_transform_args.model_dump()

        params_dict = {
            **model_args_dict,
            **data_args_dict,
            **val_transform_args_dict,  # TODO(Yutong 10/25): treat val and train separately once we fix the bug above
            "model_name": config_model,
        }
        class_ignore_index = getattr(data_args, "ignore_index", None)
        if class_ignore_index is None:
            class_ignore_index = getattr(type(data_args), "ignore_index", None)
        if class_ignore_index is not None:
            params_dict["class_ignore_index"] = class_ignore_index

        classes = getattr(data_args, "included_classes", None)
        if classes is None:
            classes = getattr(type(data_args), "included_classes", None)
        if classes is not None:
            params_dict["classes"] = classes

        normalize_args = getattr(train_transform_args, "normalize", None)
        if normalize_args is None:
            normalize_args = getattr(type(train_transform_args), "normalize", None)
        if normalize_args is not None:
            params_dict["image_normalize"] = normalize_args.model_dump()

        normalize_args = getattr(val_transform_args, "normalize", None)
        if normalize_args is None:
            normalize_args = getattr(type(val_transform_args), "normalize", None)
        if normalize_args is not None:
            params_dict["image_normalize"] = normalize_args.model_dump()

        # NOTE(Yutong 10/25): backbone_args is not actually passed into the EoMT models.
        if "backbone_args" in model_init_args:
            params_dict["backbone_args"] = model_init_args["backbone_args"]

        skipped_checkpoint_mismatch_keys = (
            {"classes", "class_ignore_index", "image_size"}
            if self.mode == "finetune"
            else set()
        )
        mismatched: dict[str, tuple[Any, Any]] = {}
        _missing = object()
        for key, checkpoint_value in model_init_args.items():
            if key in skipped_checkpoint_mismatch_keys:
                continue

            current_value = params_dict.get(key, _missing)
            if current_value is _missing:
                # Key not exposed in current config; defaults were used.
                continue
            if current_value != checkpoint_value:
                if self.mode == "resume":
                    if key in {"classes", "image_size", "image_normalize"}:
                        base_message = (
                            "Cannot resume the interrupted run because the checkpoint was saved "
                            f"with different {key}. Checkpoint {key}={checkpoint_value}, "
                            f"current {key}={current_value}."
                        )
                        raise ValueError(
                            f"{base_message} Please align the setting with the original run."
                        )

                mismatched[key] = (
                    checkpoint_value,
                    current_value,
                )

        if mismatched:
            mismatch_details = ", ".join(
                f"{key} (checkpoint={checkpoint_value}, current={current_value})"
                for key, (checkpoint_value, current_value) in sorted(mismatched.items())
            )
            logger.warning(
                "The checkpoint was created with different `model_args` values. "
                f"Mismatched keys: {mismatch_details}. Using the parameters "
                "stored in the checkpoint instead."
            )

        model_args = _merge(type(model_args), model_args, model_init_args)
        data_args = _merge(type(data_args), data_args, model_init_args)

        # TODO(Yutong 10/25): update train_transform_args separately once we fix the bug above
        # train_transform_args = _merge(
        #     type(train_transform_args), train_transform_args, model_init_args
        # )

        val_transform_args = _merge(
            type(val_transform_args), val_transform_args, model_init_args
        )

        return MetadataMergeResult(
            model_args=model_args,
            data_args=data_args,
            train_transform_args=train_transform_args,
            val_transform_args=val_transform_args,
        )

    def restore_training_state(
        self,
        *,
        state: TrainTaskState,
        reuse_class_head: bool,
    ) -> None:
        """Restore model, optimizer, and scheduler state from the checkpoint.

        Args:
            state: Training state container to populate with checkpoint data.
            reuse_class_head: Whether to keep class-specific layers when fine-tuning.

        Raises:
            ValueError: If expected components are missing from the checkpoint.
        """
        train_model = state["train_model"]
        optimizer = state["optimizer"]
        scheduler = state["scheduler"]
        train_dataloader = state["train_dataloader"]

        train_model_grads = {
            n: p.requires_grad for n, p in train_model.named_parameters()
        }
        train_model_trainings = {n: m.training for n, m in train_model.named_modules()}

        if self.train_model_state is None:
            raise ValueError(
                f"Checkpoint file '{self.path}' does not contain 'train_model'."
            )

        logger.info(f"Loading checkpoint from '{self.path}'")

        if self.mode == "finetune":
            train_model_state_keys = set(train_model.state_dict().keys())
            if reuse_class_head:
                incompatible = train_model.load_state_dict(
                    self.train_model_state, strict=False
                )
            else:
                class_head_keys = {
                    key
                    for key in train_model_state_keys
                    if key.startswith("class_head") or ".class_head" in key
                }
                criterion_keys = {
                    key
                    for key in train_model_state_keys
                    if "criterion.empty_weight" in key
                }
                checkpoint_keys_to_skip = class_head_keys | criterion_keys
                if checkpoint_keys_to_skip:
                    logger.debug(
                        "Skipping class-dependent parameters from checkpoint: %s",
                        sorted(checkpoint_keys_to_skip),
                    )
                filtered_state = {
                    key: value
                    for key, value in self.train_model_state.items()
                    if key not in checkpoint_keys_to_skip
                }
                incompatible = train_model.load_state_dict(filtered_state, strict=False)

            if reuse_class_head and incompatible.missing_keys:
                logger.warning(
                    "Missing keys after loading checkpoint: %s",
                    incompatible.missing_keys,
                )
            if incompatible.unexpected_keys:
                logger.warning(
                    "Unexpected keys after loading checkpoint: %s",
                    incompatible.unexpected_keys,
                )
        else:
            incompatible = train_model.load_state_dict(self.train_model_state)
            if incompatible.missing_keys:
                logger.warning(
                    "Missing keys after loading checkpoint: %s",
                    incompatible.missing_keys,
                )
            if incompatible.unexpected_keys:
                logger.warning(
                    "Unexpected keys after loading checkpoint: %s",
                    incompatible.unexpected_keys,
                )

            if self.optimizer_state is None or self.scheduler_state is None:
                raise ValueError(
                    f"Checkpoint file '{self.path}' does not contain optimizer or scheduler state."
                )
            optimizer.load_state_dict(self.optimizer_state)
            scheduler.load_state_dict(self.scheduler_state)

            if self.step is not None:
                state["step"] = self.step

        # Ensure that no new objects were created during loading.
        assert state["train_model"] is train_model
        assert {
            n: p.requires_grad for n, p in state["train_model"].named_parameters()
        } == train_model_grads
        assert {
            n: m.training for n, m in state["train_model"].named_modules()
        } == train_model_trainings
        if self.mode == "resume":
            assert state["optimizer"] is optimizer
            assert state["scheduler"] is scheduler
            assert state["train_dataloader"] is train_dataloader


TASK_TRAIN_MODEL_CLASSES: list[type[TrainModel]] = [
    DINOv2EoMTSemanticSegmentationTrain,
    DINOv2LinearSemanticSegmentationTrain,
    DINOv3EoMTSemanticSegmentationTrain,
]


def get_out_dir(
    fabric: Fabric,
    out: PathLike,
    resume_interrupted: bool,
    overwrite: bool,
) -> Path:
    # Use the same output directory on all ranks. This avoids issues where users
    # accidentally create different directories on each rank, for example with:
    #   out=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_global_rank_zero = fabric.broadcast(str(out))
    out_dir = Path(out_global_rank_zero)

    def check_and_create_out_dir() -> None:
        if out_dir.exists():
            if not out_dir.is_dir():
                raise ValueError(f"Output '{out_dir}' is not a directory!")

            dir_not_empty = any(out_dir.iterdir())

            if dir_not_empty and (not (resume_interrupted or overwrite)):
                raise ValueError(
                    f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite "
                    "the directory or resume_interrupted=True to resume training from "
                    "an interrupted or crashed run. "
                    "See https://docs.lightly.ai/lightly-train/usage/cli.html#resume-training "
                    "for more information on how to resume training."
                )
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

    # Create the output directory if it doesn't exist.
    with fabric.rank_zero_first():
        if fabric.global_rank == 0:
            check_and_create_out_dir()

    # Check if the output directory is on a shared filesystem. We can only check this
    # after global rank zero has created the directory.
    try:
        is_shared_filesystem = fabric_utilities.is_shared_filesystem(
            strategy=fabric.strategy, path=out_dir
        )
    except FileNotFoundError:
        # Clearly not a shared filesystem because we just created the directory.
        is_shared_filesystem = False

    # If the filesystem is not shared we have to create the output directory on every
    # node individually.
    if not is_shared_filesystem:
        with fabric.rank_zero_first(local=True):
            if fabric.local_rank == 0 and fabric.global_rank != 0:
                check_and_create_out_dir()

    return out_dir


def get_logger_args(
    steps: int,
    val_steps: int,
    logger_args: dict[str, Any] | TaskLoggerArgs | None = None,
) -> TaskLoggerArgs:
    if isinstance(logger_args, TaskLoggerArgs):
        return logger_args
    logger_args = {} if logger_args is None else logger_args
    args = validate.pydantic_model_validate(TaskLoggerArgs, logger_args)
    args.resolve_auto(steps=steps, val_steps=val_steps)
    return args


def _resolve_mlflow_run_id_for_resume(
    mlflow_args: MLFlowLoggerArgs,
) -> str | None:
    """Return the MLflow run id to resume from when resuming an interrupted run."""
    if mlflow_args.tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_args.tracking_uri)
    else:
        logger.warning(
            "No tracking_uri specified in the MLFlow logger configuration. This way we could not find the run to resume."
            "Starting a new run instead."
        )
        return None

    experiment_name = mlflow_args.experiment_name
    run_name = mlflow_args.run_name

    if not run_name:
        logger.warning(
            "Cannot resume MLflow run because no run name was specified. Please specify a `run_name` in the MLFlow logger configuration so that the metrics will continue to be logged in the same run. Starting a new run instead."
        )
        return None
    safe_run_name = run_name.replace('"', r"\"")
    filter_string = f"""
        attributes.run_name LIKE "{safe_run_name}"
        """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )

    if not runs:
        logger.warning(
            f"No MLflow runs found for experiment {experiment_name} and run name {run_name} when trying to resume. Starting a new run instead."
        )
        return None
    if len(runs) > 1:
        logger.warning(
            f"Multiple MLflow runs found for experiment {experiment_name} and run name {run_name} when trying to resume. Resuming the most recent run."
        )

    resume_run_id: str = runs[0].info.run_id

    return resume_run_id


def get_loggers(
    logger_args: TaskLoggerArgs, out: Path, resume_interrupted: bool
) -> list[FabricLogger]:
    """Get logger instances based on the provided configuration.

    All loggers are configured with the same output directory 'out'.

    Args:
        logger_args:
            Configuration for the loggers.
        out:
            Path to the output directory.
        resume_interrupted:
            Whether to resume an interrupted run. If True and an MLflow logger is
            configured, the run_id will be looked up based on the experiment_name
            and run_name and used to resume the run.
    Returns:
        List of loggers.
    """
    loggers: list[FabricLogger] = []

    if (mlflow_args := logger_args.mlflow) is not None:
        if resume_interrupted and (
            resume_run_id := _resolve_mlflow_run_id_for_resume(mlflow_args)
        ):
            if (new_run_id := mlflow_args.run_id) and new_run_id != resume_run_id:
                logger.warning(
                    f"The run_id '{new_run_id}' specified in the MLFlow logger does not match the run_id '{resume_run_id}' found when trying to resume. Using the run_id '{resume_run_id}' found with the matching `experiment_name` and `run_name` instead."
                )
            logger.debug("Resuming MLflow run with id '%s'.", resume_run_id)
            mlflow_args.run_id = resume_run_id

        logger.debug(f"Using mlflow logger with args {mlflow_args}")
        loggers.append(MLFlowLogger(save_dir=out, **mlflow_args.model_dump()))
    if logger_args.tensorboard is not None:
        logger.debug(f"Using tensorboard logger with args {logger_args.tensorboard}")
        loggers.append(
            TensorBoardLogger(save_dir=out, **logger_args.tensorboard.model_dump())
        )

    logger.debug(f"Using loggers {[log.__class__.__name__ for log in loggers]}.")
    return loggers


class PrettyFormatArgsJSONEncoder(JSONEncoder):
    """Custom JSON encoder to pretty format the output."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        try:
            return super().default(obj)
        except TypeError:
            # Return class name for objects that cannot be serialized
            return obj.__class__.__name__


def pretty_format_args(args: dict[str, Any], indent: int = 4) -> str:
    return json.dumps(
        args, indent=indent, sort_keys=True, cls=PrettyFormatArgsJSONEncoder
    )


def pretty_format_args_dict(args: dict[str, Any]) -> dict[str, Any]:
    args_str = json.dumps(args, cls=PrettyFormatArgsJSONEncoder)
    args_dict: dict[str, Any] = json.loads(args_str)
    return args_dict


def get_transform_args(
    train_model_cls: type[TrainModel],
    transform_args: dict[str, Any] | None,
    ignore_index: int | None,
) -> tuple[TaskTransformArgs, TaskTransformArgs]:
    if train_model_cls.task != "semantic_segmentation" and ignore_index is not None:
        raise ValueError(
            "`ignore_index` is only supported for semantic segmentation tasks."
        )
    transform_args = {} if transform_args is None else transform_args.copy()
    if ignore_index is not None:
        transform_args["ignore_index"] = ignore_index
    # Allows passing validation specific args via transform_args:
    # transform_args={
    #   "image_size": ..., # train only
    #   "normalize": ..., # train and val
    #   "val": {
    #       "image_size": ..., # val only
    # }
    val_args = transform_args.pop("val", {})

    train_transform_args_cls = train_model_cls.train_transform_cls.transform_args_cls
    val_transform_args_cls = train_model_cls.val_transform_cls.transform_args_cls
    train_transform_args: TaskTransformArgs
    val_transform_args: TaskTransformArgs

    train_transform_args = validate.pydantic_model_validate(
        train_transform_args_cls, transform_args
    )
    train_transform_args.resolve_auto()
    train_transform_args.resolve_incompatible()

    # Take defaults from train transform.
    val_args_dict = train_transform_args.model_dump(
        include={
            "image_size": True,
            "normalize": True,
            "ignore_index": True,
            "num_channels": True,
        }
    )
    # Overwrite with user provided val args.
    val_args_dict.update(val_args)
    val_transform_args = validate.pydantic_model_validate(
        val_transform_args_cls, val_args_dict
    )
    val_transform_args.resolve_auto()
    val_transform_args.resolve_incompatible()

    logger.debug(
        f"Resolved train transform args {pretty_format_args(train_transform_args.model_dump())}"
    )
    logger.debug(
        f"Resolved val transform args {pretty_format_args(val_transform_args.model_dump())}"
    )

    # Put back val args for logging purposes
    if val_args:
        transform_args["val"] = val_args

    return train_transform_args, val_transform_args


def get_train_transform(
    train_model_cls: type[TrainModel],
    train_transform_args: TaskTransformArgs,
) -> TaskTransform:
    return train_model_cls.train_transform_cls(transform_args=train_transform_args)


def get_val_transform(
    train_model_cls: type[TrainModel],
    val_transform_args: TaskTransformArgs,
) -> TaskTransform:
    return train_model_cls.val_transform_cls(transform_args=val_transform_args)


def get_sha256(value: Any) -> str:
    """Get the SHA256 hash of a value."""
    return hashlib.sha256(str(value).encode()).hexdigest()


def _unlink_and_ignore(path: Path) -> None:
    """Unlink a file and ignore the error if it fails.

    Errors can happen if we do not have permission to access the file.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@contextlib.contextmanager
def get_dataset_temp_mmap_path(
    fabric: Fabric,
    data: PathLike,
    out: PathLike,
) -> Generator[Path, Any, Any]:
    """Generate file in temporary directory to be used for memory-mapping the dataset.

    Creates a unique filename for the memory-mapped file based on the `out` or `data`
    arguments. We use those arguments as they are consistent across all ranks on the
    same node for the same run. Additionally, we can cache the file if required, since
    the hash directly reflects the used config.

    Use the same file on all ranks across all nodes, unless the filesystem is not shared.
    """
    if Env.LIGHTLY_TRAIN_MMAP_REUSE_FILE.value:
        # Use data as identifier to share the mmap file across multiple runs.
        # NOTE(Guarin, 09/25): Hash of data might be slow if data is a long list of
        # filenames or directories.
        identifier = Path(data).resolve()
    else:
        # Use out as identifier to create a unique mmap file for each run. We assume
        # that only one run is using a specific out directory at a time.
        identifier = Path(out).resolve()

    mmap_filepath = (cache.get_data_cache_dir() / get_sha256(identifier)).with_suffix(
        ".mmap"
    )
    mmap_filepath_broadcasted = Path(fabric.broadcast(str(mmap_filepath)))
    mmap_dirpath_broadcasted = mmap_filepath_broadcasted.parent
    ref_count_filepath_broadcasted = mmap_filepath.with_suffix(".ref_count")

    # Create the output directory if it doesn't exist.
    with fabric.rank_zero_first():
        if fabric.global_rank == 0:
            mmap_dirpath_broadcasted.mkdir(parents=True, exist_ok=True)

    # Check if the mmap directory is on a shared filesystem. We can only check this
    # after global rank zero has created the directory.
    try:
        is_shared_filesystem = fabric_utilities.is_shared_filesystem(
            strategy=fabric.strategy, path=mmap_dirpath_broadcasted
        )
    except FileNotFoundError:
        # Clearly not a shared filesystem because we just created the directory.
        is_shared_filesystem = False

    # If the filesystem is not shared we have to create the mmap file on every
    # node individually.
    if not is_shared_filesystem:
        with fabric.rank_zero_first(local=True):
            if fabric.local_rank == 0 and fabric.global_rank != 0:
                mmap_dirpath_broadcasted.mkdir(parents=True, exist_ok=True)

    try:
        # Increment reference count atomically
        _increment_ref_count(ref_count_filepath_broadcasted)

        yield mmap_filepath_broadcasted
    finally:
        # Decrement reference count and cleanup if zero
        _decrement_and_cleanup_if_zero(
            mmap_filepath_broadcasted, ref_count_filepath_broadcasted
        )


def _increment_ref_count(ref_file: Path) -> None:
    lock_file = ref_file.with_suffix(".lock")

    with FileLock(lock_file, timeout=300):
        # Ensure file exists within the lock to avoid race conditions
        ref_file.touch()
        with open(ref_file, "r+") as f:
            count = int(f.read() or "0")
            f.seek(0)
            f.write(str(count + 1))
            f.truncate()


def _decrement_and_cleanup_if_zero(mmap_file: Path, ref_file: Path) -> None:
    try:
        lock_file = ref_file.with_suffix(".lock")

        with FileLock(lock_file, timeout=300):
            with open(ref_file, "r+") as f:
                count = max(0, int(f.read() or "1") - 1)
                f.seek(0)
                f.write(str(count))
                f.truncate()

                if count <= 0 and not Env.LIGHTLY_TRAIN_MMAP_REUSE_FILE.value:
                    # Remove mmap file only if we are not reusing it and count is zero
                    _unlink_and_ignore(mmap_file)

    except (FileNotFoundError, OSError):
        pass  # Another process already cleaned up


def get_dataset_mmap_file(
    fabric: Fabric,
    items: Iterable[Mapping[str, Primitive]],
    mmap_filepath: Path,
) -> MemoryMappedSequence[Primitive]:
    """Returns memory-mapped filepaths shared across all ranks.

    Filenames are written to mmap_filepath by rank zero and read by all ranks.
    """

    # If the file already exists and we are allowed to reuse it, return it.
    if Env.LIGHTLY_TRAIN_MMAP_REUSE_FILE.value and mmap_filepath.exists():
        logger.warning(f"Reusing existing memory-mapped file '{mmap_filepath}'.")
        return MemoryMappedSequence.from_file(mmap_filepath=mmap_filepath)

    # Check if the mmap file is on a shared filesystem.
    try:
        is_shared_filesystem = fabric_utilities.is_shared_filesystem(
            strategy=fabric.strategy, path=mmap_filepath.parent
        )
    except FileNotFoundError:
        # Clearly not a shared filesystem because we just created the parent directory.
        is_shared_filesystem = False

    # If the filesystem is not shared we have to create the mmap file on every
    # node individually.
    with fabric.rank_zero_first(local=True):
        if (fabric.global_rank == 0) or (
            not is_shared_filesystem and fabric.local_rank == 0
        ):
            memory_mapped_sequence.write_items_to_file(
                items=items,
                mmap_filepath=mmap_filepath,
            )

    return MemoryMappedSequence.from_file(mmap_filepath=mmap_filepath)


def get_dataset(
    fabric: Fabric,
    dataset_args: MaskSemanticSegmentationDatasetArgs,
    transform: TaskTransform,
    mmap_filepath: Path,
) -> TaskDataset:
    image_info = dataset_args.list_image_info()

    dataset_cls = dataset_args.get_dataset_cls()
    # TODO(Guarin, 08/25): Relax this when we add object detection.
    assert isinstance(transform, SemanticSegmentationTransform)
    return dataset_cls(
        dataset_args=dataset_args,
        image_info=get_dataset_mmap_file(
            fabric=fabric,
            items=image_info,
            mmap_filepath=mmap_filepath,
        ),
        transform=transform,
    )


def get_train_dataloader(
    fabric: Fabric,
    dataset: TaskDataset,
    transform_args: TaskTransformArgs,
    batch_size: int,
    num_workers: int,
    loader_args: dict[str, Any] | None = None,
) -> DataLoader[TaskDatasetItem]:
    timeout = Env.LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0
    # TODO(Guarin, 07/25): Persistent workers by default?
    collate_fn = dataset.batch_collate_fn_cls(
        split="train", transform_args=transform_args
    )
    dataloader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size // fabric.world_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        timeout=timeout,
        collate_fn=collate_fn,
    )
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        dataloader_kwargs.update(**loader_args)
    dataloader = DataLoader(**dataloader_kwargs)
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value,no-any-return]


def get_val_dataloader(
    fabric: Fabric,
    dataset: TaskDataset,
    transform_args: TaskTransformArgs,
    batch_size: int,
    num_workers: int,
    loader_args: dict[str, Any] | None = None,
) -> DataLoader[TaskDatasetItem]:
    timeout = Env.LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0
    collate_fn = dataset.batch_collate_fn_cls(
        split="val", transform_args=transform_args
    )
    dataloader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size // fabric.world_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        timeout=timeout,
        collate_fn=collate_fn,
    )
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        dataloader_kwargs.update(**loader_args)
    dataloader = DataLoader(**dataloader_kwargs)
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value,no-any-return]


def get_steps(steps: int | Literal["auto"], default_steps: int) -> int:
    return default_steps if steps == "auto" else steps


def get_train_model_cls(model_name: str) -> type[TrainModel]:
    for train_model_cls in TASK_TRAIN_MODEL_CLASSES:
        if train_model_cls.task_model_cls.is_supported_model(model_name):
            return train_model_cls
    raise ValueError(f"Unsupported model name '{model_name}'.")


def get_train_model_args(
    model_args: dict[str, Any] | TrainModelArgs | None,
    model_args_cls: type[TrainModelArgs],
    total_steps: int,
    model_name: str,
) -> TrainModelArgs:
    if isinstance(model_args, TrainModelArgs):
        return model_args
    model_args = {} if model_args is None else model_args
    args = validate.pydantic_model_validate(model_args_cls, model_args)
    args.resolve_auto(total_steps=total_steps, model_name=model_name)
    return args


def log_step(
    split: Literal["train", "val"], step: int, max_steps: int, log_dict: dict[str, Any]
) -> None:
    split_cap = split.capitalize()
    name_to_display_name = {
        "train_loss": "Train Loss",
        "train_metric/miou": "Train mIoU",
        "val_loss": "Val Loss",
        "val_metric/miou": "Val mIoU",
    }
    parts = [
        f"{split_cap} Step {step + 1}/{max_steps}",
    ]
    for name, value in log_dict.items():
        if name in name_to_display_name:
            parts.append(f"{name_to_display_name[name]}: {value:.4f}")
    line = " | ".join(parts)
    logger.info(line)


def compute_metrics(log_dict: dict[str, Any]) -> dict[str, Any]:
    # Lazy import because torchmetrics is optional dependency.
    from torchmetrics import Metric

    metrics = {}
    for name, value in log_dict.items():
        if isinstance(value, Metric):
            value = value.compute()
        if isinstance(value, Tensor) and value.numel() > 1:
            for i, v in enumerate(value):
                metrics[f"{name}_{i}"] = v.item()
        if isinstance(value, dict):
            for class_name, class_value in value.items():
                metrics[f"{name}{class_name}"] = class_value.item()
        else:
            metrics[name] = value
    return metrics


def reset_metrics(log_dict: dict[str, Any]) -> None:
    # Lazy import because torchmetrics is optional dependency.
    from torchmetrics import Metric

    for value in log_dict.values():
        if isinstance(value, Metric):
            value.reset()


def get_save_checkpoint_args(
    checkpoint_args: dict[str, Any] | TaskSaveCheckpointArgs | None,
) -> TaskSaveCheckpointArgs:
    if isinstance(checkpoint_args, TaskSaveCheckpointArgs):
        return checkpoint_args
    checkpoint_args = {} if checkpoint_args is None else checkpoint_args
    args = validate.pydantic_model_validate(TaskSaveCheckpointArgs, checkpoint_args)
    return args


def get_checkpoint_path(
    out_dir: PathLike, best_or_last: Literal["best", "last"]
) -> Path:
    out_dir = Path(out_dir).resolve()
    ckpt_path = out_dir / "checkpoints" / f"{best_or_last}.ckpt"
    return ckpt_path


def save_checkpoint(
    fabric: Fabric,
    out_dir: Path,
    state: TrainTaskState,
    best_or_last: Literal["best", "last"],
) -> None:
    ckpt_path = get_checkpoint_path(out_dir=out_dir, best_or_last=best_or_last)

    logger.info(f"Saving the {best_or_last} checkpoint to '{ckpt_path}'")
    fabric.save(path=ckpt_path, state=state)  # type: ignore[arg-type]


def get_exported_model_path(
    out_dir: PathLike, best_or_last: Literal["best", "last"]
) -> Path:
    out_dir = Path(out_dir).resolve()
    model_path = out_dir / "exported_models" / f"exported_{best_or_last}.pt"
    return model_path


def export_model(
    out_dir: Path, model_dict: dict[str, Any], best_or_last: Literal["best", "last"]
) -> None:
    model_path = get_exported_model_path(out_dir=out_dir, best_or_last=best_or_last)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting the {best_or_last} model to '{model_path}'")
    torch.save(model_dict, model_path)
