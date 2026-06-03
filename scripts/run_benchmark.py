import lightly_train

model = lightly_train.load_model("exported_best.pt", device="cpu")

structured_benchmark_results = lightly_train.benchmark_object_detection(
    out="my_eval_run/eval_run0",
    data={
        "format": "coco",
        "train": {
            "annotations": "/home/simon/lighlty/lightly-train/datasets/coco-2017/annotations/instances_train2017.json",
            "images": "/home/simon/lighlty/lightly-train/datasets/coco-2017/val2017",
        },
        "val": {
            "annotations": "/home/simon/lighlty/lightly-train/datasets/coco-2017/annotations/instances_val2017.json",
            "images": "/home/simon/lighlty/lightly-train/datasets/coco-2017/val2017"
        },
    },
    model=model,
    num_workers=10,
    #backend_args={
    #    "format": "tensorrt",
    #},
    #backend_args={
    #    "format": "onnx",
    #    "provider": "tensorrt",
    #    "export_args": {},  # forwarded to model.export_onnx
    #},
    backend_args={"format": "torch"},
    debug=True,
    #backend_args={"format": "onnx"},
    batch_size=8,
    #steps=4,
    warmup_steps=0,
    overwrite=True,
    metric_args={
        "detection_threshold": 0.6,
        "top_k": 3,
    },
)
