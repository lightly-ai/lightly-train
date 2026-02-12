import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision(class_metrics=True)

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

metric.update(preds, targets)
result = metric.compute()

print("Keys:", list(result.keys()))
print("map_per_class:", result.get("map_per_class"))
print(
    "shape:",
    result.get("map_per_class").shape
    if result.get("map_per_class") is not None
    else None,
)
print(
    "ndim:",
    result.get("map_per_class").ndim
    if result.get("map_per_class") is not None
    else None,
)
