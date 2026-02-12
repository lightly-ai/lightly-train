import torch

from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetricArgs

metric_args = ObjectDetectionTaskMetricArgs()
detection_task_metric = metric_args.get_metrics(
    prefix="val_metric/",
    class_names=["cat__type_a", "dog__breed__b", "bird"],
    log_classwise=True,
    classwise_metric_args=None,
)

# Detection format with multiple predictions
preds = [
    {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 1]),
    }
]
targets = [
    {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
        "labels": torch.tensor([0, 1]),
    }
]

detection_task_metric.update(preds, targets)

# Add debugging
print("Computing metrics...")
for key, metric in detection_task_metric.metrics_classwise.items():
    print(f"\nMetric key: {key}")
    metric_result = metric.compute()
    print(f"Result type: {type(metric_result)}")
    if isinstance(metric_result, dict):
        print(f"Result keys: {list(metric_result.keys())}")
        map_per_class = metric_result.get("map_per_class")
        if map_per_class is not None:
            print(f"map_per_class: {map_per_class}")
            print(f"map_per_class type: {type(map_per_class)}")
            print(f"map_per_class is Tensor: {isinstance(map_per_class, torch.Tensor)}")
            if isinstance(map_per_class, torch.Tensor):
                print(f"map_per_class shape: {map_per_class.shape}")
                print(f"map_per_class ndim: {map_per_class.ndim}")
                print(f"len(class_names): {len(detection_task_metric.class_names)}")
                print(
                    f"does length match: {map_per_class.ndim == 1 and len(map_per_class) == len(detection_task_metric.class_names)}"
                )

result = detection_task_metric.compute()
print("\nFinal result keys:", list(result.keys()))
print("\nClasswise keys:", [k for k in result.keys() if "classwise" in k])
