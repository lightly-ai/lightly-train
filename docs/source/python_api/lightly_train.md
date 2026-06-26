(lightly-train)=

# lightly_train

Documentation of the public API of the `lightly_train` package.

## Functions

```{eval-rst}

.. automodule:: lightly_train
    :members: benchmark_object_detection, embed, export, export_onnx, list_methods, list_models, load_model, pretrain, train, train_image_classification, train_instance_segmentation, train_object_detection, train_panoptic_segmentation, train_semantic_segmentation

```

## Models

```{eval-rst}

.. autoclass:: lightly_train._task_models.image_classification.task_model.ImageClassification
    :members: export_onnx, export_tensorrt, predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_eomt_instance_segmentation.task_model.DINOv2EoMTInstanceSegmentation
    :members: export_onnx, export_tensorrt, predict, predict_sahi
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_eomt_instance_segmentation.task_model.DINOv3EoMTInstanceSegmentation
    :members: export_onnx, export_tensorrt, predict, predict_sahi
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetection
    :members: export_onnx, export_tensorrt, predict, predict_sahi
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.picodet_object_detection.task_model.PicoDetObjectDetection
    :members: export_onnx, export_tensorrt, predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_eomt_panoptic_segmentation.task_model.DINOv2EoMTPanopticSegmentation
    :members: export_onnx, export_tensorrt, predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentation
    :members: export_onnx, export_tensorrt, predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_eomt_semantic_segmentation.task_model.DINOv2EoMTSemanticSegmentation
    :members: export_onnx, export_tensorrt, predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model.DINOv3EoMTSemanticSegmentation
    :members: export_onnx, export_tensorrt, predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_dav2_relative_depth_estimation.task_model.DepthAnythingV2RelativeDepthEstimation
    :members: predict, predict_batch
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_dav2_metric_depth_estimation.task_model.DepthAnythingV2MetricDepthEstimation
    :members: predict, predict_batch
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_dav3_relative_depth_estimation.task_model.DepthAnythingV3RelativeDepthEstimation
    :members: predict, predict_batch
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_dav3_metric_depth_estimation.task_model.DepthAnythingV3MetricDepthEstimation
    :members: predict, predict_batch
    :exclude-members: __init__, __new__

```
