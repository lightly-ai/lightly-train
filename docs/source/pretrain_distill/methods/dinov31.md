(methods-dinov31)=

# DINOv31

DINOv31 post-trains ("continues") a DINOv2-pretrained backbone with the full DINOv2
objective (DINO + iBOT + KoLeo) plus an auxiliary Patch Kernel Alignment (PaKA / CKA)
loss that aligns the relational structure of student and teacher dense patch tokens. It
is a thin subclass of {ref}`methods-dinov2` and is aimed at improving dense
(patch-level) features for downstream segmentation and detection. See the
[PaKA paper](https://arxiv.org/abs/2509.05606) for details.

```{seealso}
DINOv31 is a DINOv2 post-training method: start from a DINOv2 checkpoint
(see {ref}`methods-dinov2`) and continue training with PaKA.
```

## Use DINOv31 in LightlyTrain

````{tab} Python
```python
import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/my_experiment",
        data="my_data_dir",
        model="dinov2/vits14",
        method="dinov31",
        checkpoint="dinov2_vits14.ckpt",  # DINOv2 init; paka_head keys tolerated
    )
```
````

````{tab} Command Line
```bash
lightly-train pretrain \
    out=out/my_experiment \
    data=my_data_dir \
    model="dinov2/vits14" \
    method="dinov31" \
    checkpoint="dinov2_vits14.ckpt"
```
````

## Default Method Arguments

The following are the default method arguments for DINOv31. To learn how you can
override these settings, see {ref}`method-args`.

````{dropdown} Default Method Arguments
```{include} _auto/dinov31_method_args.md
```
````

## Default Image Transform Arguments

The following are the default transform arguments for DINOv31. To learn how you can
override these settings, see {ref}`method-transform-args`.

````{dropdown} Default Image Transforms
```{include} _auto/dinov31_transform_args.md
```
````
