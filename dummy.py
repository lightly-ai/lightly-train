import lightly_train._commands.train_task as tt

if __name__ == "__main__":
    tt.train_task(
        out="out/finetune",
        data={
            "train": {
                "images": "dummy_data/train_images",
                "masks": "dummy_data/train_masks",
            },
            "val": {
                "images": "dummy_data/val_images",
                "masks": "dummy_data/val_masks",
            },
            "classes": {
                0: "background",
                1: "car",
            },
        },
        model="dinov2_vit/_vit_test14",
        task="semantic_segmentation",
        overwrite=True,
        task_args={
            "num_joint_blocks": 1,  # Reduce joint blocks for _vit_test14
        },
        steps=2,
        accelerator="cpu",
    )
