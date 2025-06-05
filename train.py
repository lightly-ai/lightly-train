import os
import math
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from src import lightly_train
from src.lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
import argparse
from wandb_utils import init_wandb
import pathlib


def main(args):
    # Create the output dir if needed
    output_dir = pathlib.Path(args.out)
    output_dir.mkdir(exist_ok=True)

    # Set up wandb
    name = args.out.split("/")[-1]
    init_wandb(args, name=name)

    model = DINOv2ViTPackage.get_model("vits16")

    my_transform_args = {
        "local_view": {
            "view_size": (96, 96)
        }
    }

    lr = 0.002 #  * math.sqrt(2)
    print(f"lr: {lr}")
    lightly_train.train(
        out=args.out,
        data="/datasets/imagenet/train",
        # data="/datasets/coco/images/val2017",
        model=model,
        method="dinov2",
        epochs=50,
        batch_size=2048,
        # batch_size=512, # Change here
        # overwrite=True,
        resume=True,
        # devices=1, # Change here
        transform_args=my_transform_args,
        optim_args={"lr": lr},
        precision="bf16-mixed",
        method_args={
            "ibot_separate_head": False,
            "dino_loss_weight":  1.0,
            "ibot_loss_weight":  0.0,
            "koleo_loss_weight": 0.0,
            "momentum_end": 1.0,
            "momentum_start": 0.992,
            "weight_decay_end": 0.4,
            "centering": "softmax",
            "student_freeze_last_layer_epochs": 2,
            "warmup_teacher_temp_epochs": 60,
            "warmup_epochs": 20,
            "end_teacher_temp": 0.07,
            "start_teacher_temp": 0.04,
            "output_dim": 65536,
        },
        loggers={
            "wandb": {
                "project": args.wandb_project,
                "name": name,
                "log_model": False,              # Set to True to upload model checkpoints
            },
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="/logs/thomas/dinov2/250530_dinov2_test", help="Output directory for training logs"
    )
    parser.add_argument(
        "--wandb_dir",
        default="/home/thomas/wandb",
        type=str,
        help="Name of the Weights & Biases project.",
    )
    parser.add_argument(
        "--wandb_key_path",
        default="/home/thomas/.wandb_key",
        type=str,
        help="Path to file containing the Wandb key.", 
    )
    parser.add_argument(
        "--wandb_mode",
        default="disabled",
        type=str,
        choices=["disabled", "online", "offline"]
    )
    parser.add_argument(
        "--wandb_project",
        default="dinov2",
        type=str,
        help="Name of the Weights & Biases project.",
    )
    args = parser.parse_args()

    main(args)
