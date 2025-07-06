import lightly_train
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--fsdp", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="/datasets/imagenet1k/train",
        model="dinov2_vit/vits14",
        method="dinov2",
        method_args={
            # Only set these arguments when starting from a pretrained model
            "student_freeze_backbone_epochs": 1,  # Freeze the student backbone for 1 epoch
            "student_freeze_last_layer_epochs": 0,  # Unfreeze the student last layer
        },
        batch_size=128,
        overwrite=True,
        strategy="fsdp" if args.fsdp else "ddp",
    )