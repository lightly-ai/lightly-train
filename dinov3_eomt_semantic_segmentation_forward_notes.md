# DINOv3 EoMT Semantic Segmentation Forward Notes

This note summarizes the forward, preprocessing, and postprocessing paths in
`src/lightly_train/_task_models/dinov3_eomt_semantic_segmentation/task_model.py`.

## Notation

- `B`: number of images or crops in a batch.
- `C`: number of input channels.
- `H`, `W`: image or crop height and width.
- `K`: number of configured semantic classes, `len(self.classes)`.
- `Q`: number of EoMT query tokens, `self.num_queries`.
- `E`: DINOv3 backbone embedding dimension, `self.backbone.embed_dim`.
- `P`: DINOv3 patch size, `self.backbone.patch_size`.
- `G_h = H // P`, `G_w = W // P`: patch grid size.

The model internally predicts `K + 1` classes. The first `K` channels represent the
configured classes. The extra channel represents the ignored/unknown class. Predicted
masks drop this extra channel before `argmax`, so inference always assigns every pixel
to one of the known classes.

`self.internal_class_to_class` maps internal contiguous class IDs back to the original
dataset class IDs. It is a registered buffer with dtype `torch.int32`. Returned masks
therefore have dtype `torch.int32` and contain original dataset class IDs, not
necessarily contiguous IDs.

## High-Level Paths

There are five relevant paths:

1. `predict(image)` / `predict_batch(images)`: public Python inference path.
1. `preprocess_image(...)` and `preprocess_batch(...)`: image normalization, resizing,
   tiling, and stacking.
1. `forward_backend(...)` and `postprocess(...)`: raw crop inference plus conversion
   back to original image masks.
1. `forward(...)` / `_forward_logits(...)`: export-oriented inference path.
1. `forward_train(...)`: core model computation used by training, validation, and
   inference.

## Public Prediction Path

`predict(image)` executes:

```python
x, metadata = self.preprocess_image(image)
batch = self.preprocess_batch([x])
raw = self.forward_backend(batch)
return self.postprocess(raw, [metadata])[0]
```

`predict_batch(images)` does the same, but preprocesses every input image first, then
tiles/stacks all preprocessed images in one `preprocess_batch(...)` call.

Both methods switch the model to eval mode if it is currently in training mode and run
under `torch.no_grad()`.

## `preprocess_image`

Input:

```python
image: PathLike | PIL.Image.Image | Tensor
```

Tensor inputs must have shape:

```python
Tensor[C, H_orig, W_orig]
```

Processing:

1. `file_helpers.as_image_tensor(image)` converts the input to an image tensor.
1. The tensor is moved to the same device as the model parameters.
1. It is converted to the same dtype as the first model parameter, usually
   `torch.float32`, possibly `torch.float16`.
1. `scale=True` is used during dtype conversion, so integer image data is scaled to
   floating point image range.
1. The image is normalized with `self.image_normalize["mean"]` and
   `self.image_normalize["std"]`.
1. The image is resized so the short side equals `min(self.image_size)`, while
   preserving aspect ratio.
1. Tiling metadata is computed.

Output:

```python
x: Tensor[C, H_resized, W_resized]
metadata: dict[str, Any]
```

`x`:

- dtype: model parameter floating dtype.
- device: model device.
- values: normalized image values.
- shape: one side equals `min(self.image_size)`; the long side depends on the original
  aspect ratio.

`metadata`:

```python
{
    "orig_h": H_orig,
    "orig_w": W_orig,
    "resized_h": H_resized,
    "resized_w": W_resized,
    "origins": list[(image_idx, start, end, is_tall)],
    "num_crops": int,
}
```

Important: `preprocess_image(...)` outputs are not necessarily stackable across images
because resized long sides can differ.

## `preprocess_batch`

Input:

```python
batch: Sequence[Tensor[C, H_i, W_i]]
```

The tensors are expected to already be normalized and resized by
`preprocess_image(...)`.

Processing:

1. Each image is split into square crops using `tile(...)`.
1. Crops are stacked into one tensor.

Output:

```python
Tensor[sum_i N_i, C, S, S]
```

Where:

- `S = min(H_i, W_i)`, normally `min(self.image_size)`.
- `N_i = ceil(max(H_i, W_i) / S)`.

## `tile`

Input:

```python
images: list[Tensor[C, H, W]] | Tensor[B, C, H, W]
```

For each image:

```python
crop_size = min(H, W)
num_crops = ceil(max(H, W) / crop_size)
```

Each crop has shape:

```python
Tensor[C, crop_size, crop_size]
```

If the image is tall, crops slice along height:

```python
image[:, start:end, :]
```

If the image is wide or square, crops slice along width:

```python
image[:, :, start:end]
```

Output:

```python
crops: list[Tensor[C, crop_size, crop_size]]
origins: list[(image_index, start, end, is_tall)]
```

Overlaps are introduced when the long side is not divisible by `crop_size`.

## `untile`

Input:

```python
crop_logits: Tensor[N_crops, channels, crop_h, crop_w]
origins: list[(image_index, start, end, is_tall)]
image_sizes: list[(H_img, W_img)]
```

Processing:

1. Creates zero tensors for logit sums and overlap counts.
1. Adds each crop into its corresponding image canvas.
1. Divides by overlap counts.

Output:

```python
list[Tensor[channels, H_img, W_img]]
```

The result is an averaged reconstruction over overlapping crop regions.

## `forward_backend`

This is the raw inference forward for already preprocessed square crops.

Input:

```python
x: Tensor[B_crops, C, S, S]
```

Expected properties:

- dtype: floating point.
- values: normalized image values.
- shape: square crops.

Processing:

```python
mask_logits_per_layer, class_logits_per_layer = self.forward_train(
    x,
    return_logits_per_layer=False,
)
mask_logits = mask_logits_per_layer[-1]
class_logits = class_logits_per_layer[-1]
mask_logits = F.interpolate(mask_logits, (S, S), mode="bilinear")
```

Output:

```python
mask_logits: Tensor[B_crops, Q, S, S]
class_logits: Tensor[B_crops, Q, K + 1]
```

`mask_logits` are query-specific spatial mask logits, resized to the input crop size.
`class_logits` are query-specific class logits over the `K + 1` semantic classes.

## `to_per_pixel_logits_semantic`

Input:

```python
mask_logits: Tensor[B, Q, H, W]
class_logits: Tensor[B, Q, K + 1]
```

Processing:

```python
torch.einsum(
    "bqhw,bqc->bchw",
    mask_logits.sigmoid(),
    class_logits.softmax(dim=-1),
)
```

Output:

```python
Tensor[B, K + 1, H, W]
```

These are per-pixel semantic class scores. Despite the method and variable names, after
`sigmoid` and `softmax` they are not raw logits anymore. They combine:

- query mask probabilities: `Tensor[B, Q, H, W]`;
- query class probabilities: `Tensor[B, Q, K + 1]`.

## `postprocess`

Input:

```python
raw_outputs: tuple[
    Tensor[sum_i N_i, Q, S, S],
    Tensor[sum_i N_i, Q, K + 1],
]
metadata: Sequence[dict[str, Any]]
```

Processing:

1. Converts query mask/class outputs to per-pixel semantic scores:

   ```python
   crop_logits_all: Tensor[sum_i N_i, K + 1, S, S]
   ```

1. Splits crop scores by image using `meta["num_crops"]`.

1. Untiles each image's crop scores back to resized image size:

   ```python
   Tensor[K + 1, H_resized, W_resized]
   ```

1. Drops the extra ignored/unknown channel:

   ```python
   Tensor[K, H_resized, W_resized]
   ```

1. Interpolates to original image size:

   ```python
   Tensor[1, K, H_orig, W_orig]
   ```

1. Takes `argmax(dim=1)`:

   ```python
   Tensor[1, H_orig, W_orig]
   ```

1. Maps internal class IDs back to original class IDs:

   ```python
   Tensor[1, H_orig, W_orig], dtype=torch.int32
   ```

Output:

```python
list[Tensor[H_orig, W_orig]]
```

Each returned tensor is a semantic segmentation mask in original image resolution.

## Direct `forward`

`forward(x)` is mainly used for ONNX/export-style inference.

Input:

```python
x: Tensor[B, C, H, W]
```

Important: this path does not call `preprocess_image(...)`. The caller must pass
model-ready tensors, meaning correct dtype/range, normalization, shape, and device.

Processing:

```python
logits = self._forward_logits(x)  # Tensor[B, K + 1, H, W]
logits = logits[:, :-1]           # Tensor[B, K, H, W]
masks = logits.argmax(dim=1)      # Tensor[B, H, W]
masks = self.internal_class_to_class[masks]
```

Output:

```python
masks: Tensor[B, H, W], dtype=torch.int32
logits: Tensor[B, K, H, W], floating point
```

The returned `logits` are semantic class scores after query mask sigmoid and query class
softmax composition. They are not pure raw logits.

## `_forward_logits`

Input:

```python
x: Tensor[B, C, H, W]
```

Processing:

1. If running in ONNX export and `H != W`, raises `ValueError`.
1. If running in ONNX export or `H == W`, uses `x` directly as crops.
1. Otherwise, tiles non-square images into square crops and stacks them.
1. Calls `forward_train(crops, return_logits_per_layer=False)`.
1. Interpolates query mask logits to crop size.
1. Converts query outputs to per-pixel semantic scores.
1. Untiles back to `(H, W)` if tiling was used.

Output:

```python
Tensor[B, K + 1, H, W]
```

For ONNX export, only square images are supported because the dynamic Python
tiling/untilling path is not export-compatible.

## Core `forward_train`

This is the core computation used by training, validation, `forward_backend`, and
`_forward_logits`.

Input:

```python
x: Tensor[B, C, H, W]
```

Expected properties:

- floating point dtype;
- normalized image values;
- no preprocessing is done inside this method.

The patch grid is:

```python
grid_size = (H // P, W // P)
```

The method follows the DINOv3 patch embedding behavior, which means pixels outside a
full patch grid can be dropped if `H` or `W` is not divisible by `P`.

After:

```python
x, image_size = self.backbone.prepare_tokens_with_masks(x)
```

the approximate token tensor shape is:

```python
Tensor[B, 1 + S_tokens + G_h * G_w, E]
```

Where:

- `1` is the class token.
- `S_tokens = self.backbone.n_storage_tokens`.
- `G_h * G_w` are patch tokens.

At the first joint block, query tokens are prepended:

```python
queries: Tensor[B, Q, E]
x: Tensor[B, Q + 1 + S_tokens + G_h * G_w, E]
```

If `return_logits_per_layer=False`, only the final prediction is returned:

```python
mask_logits_per_layer: list[Tensor[B, Q, H_mask, W_mask]]  # length 1
class_logits_per_layer: list[Tensor[B, Q, K + 1]]          # length 1
```

If `return_logits_per_layer=True`, the model predicts once before each joint block and
once after the final block:

```python
len(mask_logits_per_layer) == self.num_joint_blocks + 1
len(class_logits_per_layer) == self.num_joint_blocks + 1
```

Each element has shape:

```python
mask_logits: Tensor[B, Q, H_mask, W_mask]
class_logits: Tensor[B, Q, K + 1]
```

`H_mask` and `W_mask` depend on `self.upscale`. With default
`fix_num_upscale_blocks=True`, there are two `ScaleBlock`s. Each `ScaleBlock` uses
stride-2 transposed convolution, so spatial size is approximately:

```python
H_mask = (H // P) * 4
W_mask = (W // P) * 4
```

For a `P=16` backbone, mask logits are usually at one quarter of input resolution before
later interpolation.

## `_predict`

Input:

```python
x: Tensor[B, Q + 1 + S_tokens + G_h * G_w, E]
grid_size: tuple[int, int] = (G_h, G_w)
```

Processing:

```python
q = x[:, :Q, :]                         # Tensor[B, Q, E]
class_logits = self.class_head(q)       # Tensor[B, Q, K + 1]

patch_tokens = x[:, Q + 1 + S_tokens:, :]
patch_map = patch_tokens.transpose(1, 2).reshape(B, E, G_h, G_w)

mask_embeddings = self.mask_head(q)     # Tensor[B, Q, E]
features = self.upscale(patch_map)      # Tensor[B, E, H_mask, W_mask]

mask_logits = torch.einsum(
    "bqc,bchw->bqhw",
    mask_embeddings,
    features,
)
```

Output:

```python
mask_logits: Tensor[B, Q, H_mask, W_mask]
class_logits: Tensor[B, Q, K + 1]
```

Each query predicts both a semantic class distribution and a spatial mask.

## Training Step

During training, the semantic segmentation transform applies resize/crop, augmentation,
normalization, and tensor conversion. The collate function stacks training images and
masks because training crops have fixed shape.

Training batch shapes:

```python
images: Tensor[B, C, H, W]
masks: Tensor[B, H, W]
binary_masks: list[dict[str, Tensor]]
```

`binary_masks` entries contain binary target masks and labels for loss computation. The
exact number of target masks can vary per image.

The training step calls:

```python
mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(
    images,
    return_logits_per_layer=True,
)
```

Loss is computed for each returned layer: all joint block predictions and the final
prediction.

For training metrics, only the final layer is used:

```python
mask_logits = mask_logits_per_layer[-1]
class_logits = class_logits_per_layer[-1]
mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
scores = self.model.to_per_pixel_logits_semantic(mask_logits, class_logits)
scores = scores[:, :-1]  # Drop ignored/unknown class
```

The resulting metric tensor has shape:

```python
Tensor[B, K, H, W]
```

## Validation Step

Validation batches are not stacked by the collate function because validation images can
have different shapes.

Validation batch shapes:

```python
images: list[Tensor[C, H_i, W_i]]
masks: list[Tensor[H_i, W_i]]
binary_masks: list[dict[str, Tensor]]
```

Validation explicitly tiles the images:

```python
crops_list, origins = self.model.tile(images)
crops = torch.stack(crops_list)
```

Then:

```python
mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(
    crops,
    return_logits_per_layer=True,
)
```

For the final layer, validation:

1. Interpolates mask logits to crop size.

1. Converts query outputs to per-pixel semantic scores.

1. Drops the extra ignored/unknown class channel.

1. Untiles crop scores back to each validation image size.

1. Updates metrics with tensors shaped:

   ```python
   Tensor[1, K, H_i, W_i]
   ```

Loss is still computed for all returned layers.

## Attention Mask During Training

When all of the following are true:

- `return_logits_per_layer=True`;
- current block is one of the joint blocks;
- `self.training` is true;

the model builds a boolean attention mask from intermediate mask predictions.

Shape:

```python
attn_mask: Tensor[B, N_tokens, N_tokens], dtype=torch.bool
```

Where:

```python
N_tokens = Q + 1 + S_tokens + G_h * G_w
```

The mask controls query-to-patch attention in the joint blocks. It is annealed by
`self.attn_mask_probs`.

Inference does not use this attention masking because inference calls
`forward_train(..., return_logits_per_layer=False)`.

## Dataloader Transform Difference

Training:

- Images are randomly transformed/cropped to fixed shape.

- Collate stacks them:

  ```python
  image: Tensor[B, C, H, W]
  mask: Tensor[B, H, W]
  ```

Validation:

- Images can have different shapes.

- Collate keeps them as lists:

  ```python
  image: list[Tensor[C, H_i, W_i]]
  mask: list[Tensor[H_i, W_i]]
  ```

This difference is why validation uses explicit tiling before `forward_train`, while
training can forward the stacked tensor directly.
