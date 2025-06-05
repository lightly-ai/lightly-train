import argparse
import pathlib

from src import lightly_train
from src.lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
from wandb_utils import init_wandb


def main(args):
    output_dir = pathlib.Path(args.out)
    output_dir.mkdir(exist_ok=True)

    name = args.out.split("/")[-1]
    init_wandb(args, name=name)

    model = DINOv2ViTPackage.get_model(args.arch)

    transform_args = {
        "local_view": {
            "view_size": tuple(map(int, args.local_view_size.split(',')))
        }
    }

    lr = args.lr
    print(f"lr: {lr}")

    lightly_train.train(
        out=args.out,
        data=args.data,
        model=model,
        method="dinov2",
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        transform_args=transform_args,
        optim_args={"lr": lr},
        precision=args.precision,
        overwrite=True,
        loader_args={"num_workers": 15},
        # devices=1,
        method_args={
            "ibot_separate_head": args.ibot_separate_head,
            "dino_loss_weight": args.dino_loss_weight,
            "ibot_loss_weight": args.ibot_loss_weight,
            "koleo_loss_weight": args.koleo_loss_weight,
            "momentum_end": args.momentum_end,
            "momentum_start": args.momentum_start,
            "weight_decay_end": args.weight_decay_end,
            "centering": args.centering,
            "student_freeze_last_layer_epochs": args.student_freeze_last_layer_epochs,
            "warmup_teacher_temp_epochs": args.warmup_teacher_temp_epochs,
            "warmup_epochs": args.warmup_epochs,
            "end_teacher_temp": args.end_teacher_temp,
            "start_teacher_temp": args.start_teacher_temp,
            "output_dim": args.output_dim,
        },
        loggers={
            "wandb": {
                "project": args.wandb_project,
                "name": name,
                "log_model": args.log_model,
            },
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths and metadata
    parser.add_argument("--out", type=str, default="/logs/thomas/dinov2/debug", help="Output directory for training logs")
    parser.add_argument("--data", type=str, default="/datasets/imagenet/train", help="Dataset path")
    parser.add_argument("--arch", type=str, default="vits16", help="DINOv2 architecture to use")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision format")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)

    # Method arguments
    parser.add_argument("--ibot_separate_head", action="store_true")
    parser.add_argument("--dino_loss_weight", type=float, default=1.0)
    parser.add_argument("--ibot_loss_weight", type=float, default=1.0)
    parser.add_argument("--koleo_loss_weight", type=float, default=0.1)
    parser.add_argument("--momentum_end", type=float, default=1.0)
    parser.add_argument("--momentum_start", type=float, default=0.992)
    parser.add_argument("--weight_decay_end", type=float, default=0.4)
    parser.add_argument("--centering", type=str, default="softmax")
    parser.add_argument("--student_freeze_last_layer_epochs", type=int, default=2)
    parser.add_argument("--warmup_teacher_temp_epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--end_teacher_temp", type=float, default=0.07)
    parser.add_argument("--start_teacher_temp", type=float, default=0.04)
    parser.add_argument("--output_dim", type=int, default=65536)
    parser.add_argument("--local_view_size", type=str, default="96,96", help="Size of the local view as 'H,W'")

    # Wandb logging
    parser.add_argument("--wandb_project", type=str, default="dinov2")
    parser.add_argument("--wandb_dir", type=str, default="/home/thomas/wandb")
    parser.add_argument("--wandb_key_path", type=str, default="/home/thomas/.wandb_key")
    parser.add_argument("--wandb_mode", type=str, choices=["disabled", "online", "offline"], default="disabled")
    parser.add_argument("--log_model", action="store_true", help="Upload model checkpoints to Wandb")

    args = parser.parse_args()
    main(args)
