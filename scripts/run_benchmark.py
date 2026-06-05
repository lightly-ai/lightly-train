import lightly_train

if __name__ == "__main__":
    # model = lightly_train.load_model("exported_best.pt", device="cpu")
    # model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
    # model = lightly_train.load_model("picodet-s-coco")
    model = lightly_train.load_model("dinov3/convnext-tiny-ltdetr-coco")

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
                "images": "/home/simon/lighlty/lightly-train/datasets/coco-2017/val2017",
            },
        },
        model=model,
        num_workers=0,
        # backend_args={"format": "tensorrt", "precision": "fp16", "export_args": {"onnx_args": {"dynamic_batch_size": False}}},
        # backend_args={
        #    "format": "onnx",
        #    "provider": "tensorrt",
        #    "export_args": {"dynamic_batch_size": False},
        # },
        backend_args={"format": "torch", "compile": False},
        # backend_args={"format": "onnx", "provider": "cuda", "precision": "fp32", "export_args": {"dynamic_batch_size": False}},
        # backend_args={"format": "onnx", "provider": "tensorrt", "precision": "fp16", "export_args": {"dynamic_batch_size": False}},
        # backend_args={"format": "onnx", "provider": "cpu", "export_args": {"dynamic_batch_size": False}},
        # device="cpu",
        debug=True,
        batch_size=8,
        steps=None,
        warmup_steps=0,
        overwrite=True,
        metric_args={
            "detection_threshold": 0.6,
            "top_k": 3,
        },
    )
