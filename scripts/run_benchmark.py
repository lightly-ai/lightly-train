import lightly_train

model = lightly_train.load_model("exported_best.pt")

structured_benchmark_results = lightly_train.benchmark_object_detection(
    out="my_eval_run/eval_run0",
    data={
        "format": "coco",
        "train": {
            "annotations": "/home/simon/Datasets/annotations/instances_train2017.json",
            "images": "/home/simon/Datasets/coco-2017/val2017",
        },
        "val": {
            "annotations": "/home/simon/Datasets/annotations/instances_val2017.json",
            "images": "/home/simon/Datasets/coco-2017/val2017",
        },
    },
    model=model,
    num_workers=10,
    backend_args={
        "format": "onnx",
        "export_args": {},  # forwarded to model.export_onnx
    },
    batch_size=8,
    steps=4,
    overwrite=True,
    metric_args={
        "detection_threshold": 0.6,
        "top_k": 3,
    },
)
