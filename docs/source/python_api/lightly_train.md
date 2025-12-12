(lightly-train)=

# lightly_train

Documentation of the public API of the `lightly_train` package.

## Functions

```{eval-rst}

.. automodule:: lightly_train
    :members: embed, export, export_onnx, list_methods, list_models, load_model, train, train_instance_segmentation, train_object_detection, train_panoptic_segmentation, train_semantic_segmentation

```

## Models

```{eval-rst}

.. autoclass:: lightly_train._task_models.dinov3_eomt_instance_segmentation.task_model.DINOv3EoMTInstanceSegmentation
    :members: predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_ltdetr_object_detection.task_model.DINOv2LTDETRObjectDetection
    :members: predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_ltdetr_object_detection.task_model.DINOv3LTDETRObjectDetection
    :members: predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model.DINOv3EoMTPanopticSegmentation
    :members: predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov2_eomt_semantic_segmentation.task_model.DINOv2EoMTSemanticSegmentation
    :members: predict
    :exclude-members: __init__, __new__

.. autoclass:: lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model.DINOv3EoMTSemanticSegmentation
    :members: predict
    :exclude-members: __init__, __new__

```
