import argparse
from pathlib import Path

from lightly_train import train_task
from ade20k_classes import ADE20K_DICT
import os
import mlflow
import pathlib


def get_rank():
    """Returns the rank based on environment variables (works before DDP init)."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0  # Assume single process or local run


def is_main_process():
    return get_rank() == 0


def get_or_create_run(output_dir, experiment_name, run_name=None, tracking_uri=None):
    run_file = os.path.join(output_dir, "mlflow_run_id.txt")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    if is_main_process():
        if os.path.exists(run_file):
            with open(run_file, "r") as f:
                run_id = f.read().strip()
            print(f"[Rank 0] Resuming MLflow run: {run_id}")
        else:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
            with open(run_file, "w") as f:
                f.write(run_id)
            print(f"[Rank 0] Created new MLflow run: {run_id}")
    else:
        # Wait until file exists (written by rank 0)
        while not os.path.exists(run_file):
            import time

            time.sleep(1)  # avoid busy waiting
        with open(run_file, "r") as f:
            run_id = f.read().strip()
        print(f"[Rank {get_rank()}] Loaded MLflow run ID: {run_id}")

    return run_id


def main(args):
    # Set the output dir.
    output_dir = pathlib.Path(args.out)
    output_dir.mkdir(exist_ok=True, parents=True)
    run_name = args.out.split("/")[-1]

    # Get or create the run ID.
    mlflow_args = None
    if args.use_mlflow:
        run_id = get_or_create_run(
            output_dir=args.out,
            experiment_name=args.mlflow_experiment_name,
            run_name=run_name,
            tracking_uri=args.mlflow_uri,
        )

        # Set the MLFlow args.
        mlflow_args = {
            "experiment_name": args.mlflow_experiment_name,
            "run_name": run_name,
            "tracking_uri": args.mlflow_uri,
            "run_id": run_id,
        }

    os.environ["MLFLOW_TRACKING_USERNAME"] = args.mlflow_username
    # Data args.
    data = {
        "train": {
            "images": "/datasets/ade20k/ADEChallengeData2016/images/training",
            "masks": "/datasets/ade20k/ADEChallengeData2016/annotations/training",
        },
        "val": {
            "images": "/datasets/ade20k/ADEChallengeData2016/images/validation",
            "masks": "/datasets/ade20k/ADEChallengeData2016/annotations/validation",
        },
        "classes": ADE20K_DICT,
    }

    # Task args.
    task_args = {
        "ignore_index": 255,
    }

    # Set the number of training steps.
    n_train_images = 20210
    steps_per_epoch = int(n_train_images / args.batchsize)
    total_steps = steps_per_epoch * args.epochs
    print(f"steps_per_epoch: {steps_per_epoch}.")
    print(f"total_steps: {total_steps}.")

    # Set the validation frequency.
    logger_args = {
        "val_every_num_steps": steps_per_epoch,
    }
    if mlflow_args:
        logger_args["mlflow"] = mlflow_args

    # Set the checkpoint args.
    checkpoint_args = {
        "save_every_num_steps": 10 * steps_per_epoch,
    }

    # Launch training.
    train_task(
        out=args.out,
        data=data,
        model=args.model,
        task="semantic_segmentation",
        steps=total_steps,
        batch_size=args.batchsize,
        num_workers=7,
        overwrite=True,
        resume_interrupted=args.resume_interrupted,
        task_args=task_args,
        logger_args=logger_args,
        checkpoint_args=checkpoint_args,
        # devices=[7],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Lightly task.")
    parser.add_argument(
        "--out", type=str, default="out/my_experiment_1gpu", help="Output directory path."
    )
    parser.add_argument(
        "--batchsize", type=int, default=16, help="Global batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=31, help="Number of epochs."
    )
    parser.add_argument(
        "--resume-interrupted",
        action="store_true",
        help="Resume from an interrupted run.",
    )
    parser.add_argument("--model", type=str, default="dinov2_vit/vitl14_pretrain", help="")
    parser.add_argument("--mlflow-experiment-name", type=str, default="eomt", help="MLflow experiment name")
    parser.add_argument("--mlflow-uri", type=str, default="http://compute-03-ubuntu-4x4090:5000/", help="MLflow tracking URI")
    parser.add_argument("--mlflow-username", type=str, default="thomas", help="MLflow username")
    # parser.add_argument("--use_mlflow", action="store_true")
    parser.add_argument("--use_mlflow", default=False)
    return parser.parse_args()



if __name__ == "__main__":
    # Parse the args.
    args = parse_args()

    # Launch.
    main(args)
