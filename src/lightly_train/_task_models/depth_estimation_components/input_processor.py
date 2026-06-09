#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Input processor for Depth Anything 3.

Converts a list of images into a single model-ready batch tensor, processing each
image sequentially. The square center-crop step is omitted for "*crop" methods.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torchvision.transforms.v2.functional as tv_functional
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)

# (image_tensor, (H, W), intrinsic, extrinsic) produced for a single input image.
_ProcessedItem = tuple[Tensor, tuple[int, int], np.ndarray | None, np.ndarray | None]


class InputProcessor:
    """Prepares a batch of images for model inference.

    This processor converts a list of image file paths into a single, model-ready
    tensor. Each image is processed sequentially and the order of outputs matches
    the input order.

    Pipeline:
      1) Load image and convert to RGB
      2) Boundary resize (upper/lower bound, preserving aspect ratio)
      3) Enforce divisibility by PATCH_SIZE:
         - "*resize" methods: each dimension is rounded to nearest multiple
           (may up/downscale a few px)
         - "*crop"   methods: each dimension is floored to nearest multiple via center crop
      4) Convert to tensor and apply ImageNet normalization
      5) Stack into (1, N, 3, H, W)
    """

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)
    PATCH_SIZE = 14

    def __init__(self) -> None:
        pass

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            (tensor, extrinsics_list, intrinsics_list)
            tensor shape: (1, N, 3, H, W)
        """
        exts_list, ixts_list = self._validate_and_pack_meta(
            image, extrinsics, intrinsics
        )

        results = self._process_all(
            image=image,
            exts_list=exts_list,
            ixts_list=ixts_list,
            process_res=process_res,
            process_res_method=process_res_method,
        )

        proc_imgs, out_sizes, out_ixts, out_exts = self._unpack_results(results)
        proc_imgs, out_sizes, out_ixts = self._unify_batch_shapes(
            proc_imgs, out_sizes, out_ixts
        )

        batch_tensor = self._stack_batch(proc_imgs)
        out_exts_tensor = (
            torch.from_numpy(np.asarray(out_exts)).float()
            if out_exts is not None and out_exts[0] is not None
            else None
        )
        out_ixts_tensor = (
            torch.from_numpy(np.asarray(out_ixts)).float()
            if out_ixts is not None and out_ixts[0] is not None
            else None
        )
        return (batch_tensor, out_exts_tensor, out_ixts_tensor)

    # -----------------------------
    # __call__ helpers
    # -----------------------------
    def _validate_and_pack_meta(
        self,
        images: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> tuple[list[np.ndarray | None] | None, list[np.ndarray | None] | None]:
        if extrinsics is not None and len(extrinsics) != len(images):
            raise ValueError("Length of extrinsics must match images when provided.")
        if intrinsics is not None and len(intrinsics) != len(images):
            raise ValueError("Length of intrinsics must match images when provided.")
        exts_list = [e for e in extrinsics] if extrinsics is not None else None
        ixts_list = [k for k in intrinsics] if intrinsics is not None else None
        return exts_list, ixts_list

    def _process_all(
        self,
        *,
        image: list[np.ndarray | Image.Image | str],
        exts_list: list[np.ndarray | None] | None,
        ixts_list: list[np.ndarray | None] | None,
        process_res: int,
        process_res_method: str,
    ) -> list[_ProcessedItem]:
        results = [
            self._process_one(
                img=img,
                extrinsic=exts_list[i] if exts_list is not None else None,
                intrinsic=ixts_list[i] if ixts_list is not None else None,
                process_res=process_res,
                process_res_method=process_res_method,
            )
            for i, img in enumerate(image)
        ]
        if not results:
            raise RuntimeError(
                "No preprocessing results returned; the input image list is empty."
            )
        return results

    def _unpack_results(
        self, results: Sequence[_ProcessedItem]
    ) -> tuple[
        list[Tensor],
        list[tuple[int, int]],
        list[np.ndarray | None],
        list[np.ndarray | None],
    ]:
        """
        results: per-image (image_tensor, (H, W), intrinsic, extrinsic) tuples
        -> processed_images, out_sizes, out_intrinsics, out_extrinsics
        """
        try:
            processed_images, out_sizes, out_intrinsics, out_extrinsics = zip(*results)
        except Exception as e:
            raise RuntimeError(
                "Unexpected results structure from preprocessing: "
                f"{type(results)} / sample: {results[0]}"
            ) from e

        return (
            list(processed_images),
            list(out_sizes),
            list(out_intrinsics),
            list(out_extrinsics),
        )

    def _unify_batch_shapes(
        self,
        processed_images: list[torch.Tensor],
        out_sizes: list[tuple[int, int]],
        out_intrinsics: list[np.ndarray | None],
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]], list[np.ndarray | None]]:
        """Center-crop all tensors to the smallest H, W; adjust intrinsics' cx, cy accordingly."""
        if len(set(out_sizes)) <= 1:
            return processed_images, out_sizes, out_intrinsics

        min_h = min(h for h, _ in out_sizes)
        min_w = min(w for _, w in out_sizes)
        logger.warning(
            f"Images in batch have different sizes {out_sizes}; "
            f"center-cropping all to smallest ({min_h},{min_w})"
        )

        new_imgs: list[Tensor] = []
        new_sizes: list[tuple[int, int]] = []
        new_ixts: list[np.ndarray | None] = []
        for img_t, (H, W), K in zip(processed_images, out_sizes, out_intrinsics):
            crop_top = max(0, (H - min_h) // 2)
            crop_left = max(0, (W - min_w) // 2)
            new_imgs.append(
                tv_functional.center_crop(img_t, output_size=[min_h, min_w])
            )
            new_sizes.append((min_h, min_w))
            if K is None:
                new_ixts.append(None)
            else:
                K_adj = K.copy()
                K_adj[0, 2] -= crop_left
                K_adj[1, 2] -= crop_top
                new_ixts.append(K_adj)
        return new_imgs, new_sizes, new_ixts

    def _stack_batch(self, processed_images: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(processed_images)

    # -----------------------------
    # Per-item worker
    # -----------------------------
    def _process_one(
        self,
        img: np.ndarray | Image.Image | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
    ) -> tuple[torch.Tensor, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        # Load & remember original size
        image = self._load_image(img)
        orig_h, orig_w = image.shape[-2:]

        # Boundary resize
        image = self._resize_image(image, process_res, process_res_method)
        h, w = image.shape[-2:]
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w, h)

        # Enforce divisibility by PATCH_SIZE
        if process_res_method.endswith("resize"):
            image = self._make_divisible_by_resize(image, self.PATCH_SIZE)
            new_h, new_w = image.shape[-2:]
            intrinsic = self._resize_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        elif process_res_method.endswith("crop"):
            image = self._make_divisible_by_crop(image, self.PATCH_SIZE)
            new_h, new_w = image.shape[-2:]
            intrinsic = self._crop_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        # Convert to float tensor & normalize
        img_tensor = self._normalize_image(image)
        _, H, W = img_tensor.shape
        assert (W, H) == (w, h), (
            "Tensor size mismatch with PIL image size after processing."
        )

        # Return: (img_tensor, (H, W), intrinsic, extrinsic)
        return img_tensor, (H, W), intrinsic, extrinsic

    # -----------------------------
    # Intrinsics transforms
    # -----------------------------
    def _resize_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        # scale fx, cx by w ratio; fy, cy by h ratio
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K

    def _crop_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        crop_h = (orig_h - h) // 2
        crop_w = (orig_w - w) // 2
        K[0, 2] -= crop_w
        K[1, 2] -= crop_h
        return K

    # -----------------------------
    # I/O & normalization
    # -----------------------------
    def _load_image(self, img: np.ndarray | Image.Image | str) -> Tensor:
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            # Assume HxWxC uint8/RGB.
            pil_img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        # Decode to a uint8 (3, H, W) tensor; all transforms below stay in tensor space.
        image: Tensor = tv_functional.pil_to_tensor(pil_img)
        return image

    def _normalize_image(self, img: Tensor) -> Tensor:
        img = tv_functional.to_dtype(img, dtype=torch.float32, scale=True)
        img = tv_functional.normalize(
            img, mean=list(self.NORMALIZE_MEAN), std=list(self.NORMALIZE_STD)
        )
        return img

    # -----------------------------
    # Boundary resizing
    # -----------------------------
    def _resize_image(self, img: Tensor, target_size: int, method: str) -> Tensor:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side(img, target_size)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side(img, target_size)
        else:
            raise ValueError(f"Unsupported resize method: {method}")

    def _resize_longest_side(self, img: Tensor, target_size: int) -> Tensor:
        h, w = img.shape[-2:]
        longest = max(w, h)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return self._resize_to(img, new_h, new_w)

    def _resize_shortest_side(self, img: Tensor, target_size: int) -> Tensor:
        h, w = img.shape[-2:]
        shortest = min(w, h)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return self._resize_to(img, new_h, new_w)

    def _resize_to(self, img: Tensor, new_h: int, new_w: int) -> Tensor:
        """Resizes a ``(C, H, W)`` uint8 image to ``(new_h, new_w)``.

        Reproduces the cv2 resampling the official DA3 reference depends on without an
        OpenCV dependency: ``INTER_AREA`` when shrinking and ``INTER_CUBIC`` when
        enlarging. Both are separable, so per-axis ``(dst, src)`` weight matrices are
        built and applied along the height then the width, accumulating in float and
        rounding back to uint8 once. As in cv2, the kernel choice is joint over both axes:
        a resize that enlarges either axis uses cubic. Matches cv2 to within one uint8
        level.
        """
        h, w = img.shape[-2:]
        enlarging = new_h > h or new_w > w
        weights = self._cubic_weights if enlarging else self._area_weights
        row_weights = weights(src=h, dst=new_h, device=img.device)
        col_weights = weights(src=w, dst=new_w, device=img.device)
        x = img.to(torch.float32)
        x = torch.einsum("ph,chw->cpw", row_weights, x)
        x = torch.einsum("qw,cpw->cpq", col_weights, x)
        return x.round().clamp(min=0, max=255).to(torch.uint8)

    @staticmethod
    def _area_weights(*, src: int, dst: int, device: torch.device) -> Tensor:
        """Builds the ``(dst, src)`` cv2 ``INTER_AREA`` overlap-weight matrix for one axis.

        Output cell ``i`` spans ``[i * scale, (i + 1) * scale)`` in source coordinates,
        where ``scale = src / dst``; the weight for source pixel ``j`` is the length of
        the overlap with ``[j, j + 1)`` divided by ``scale``. A ``scale`` of 1 yields the
        identity, so an unchanged axis is passed through exactly.
        """
        scale = src / dst
        out_idx = torch.arange(dst, dtype=torch.float32, device=device)[:, None]
        src_idx = torch.arange(src, dtype=torch.float32, device=device)[None, :]
        lo = torch.maximum(out_idx * scale, src_idx)
        hi = torch.minimum((out_idx + 1) * scale, src_idx + 1)
        return (hi - lo).clamp(min=0) / scale

    @staticmethod
    def _cubic_weights(*, src: int, dst: int, device: torch.device) -> Tensor:
        """Builds the ``(dst, src)`` cv2 ``INTER_CUBIC`` weight matrix for one axis.

        Uses the Keys cubic convolution kernel with ``a = -0.75`` (cv2's constant) and
        half-pixel sample centers. Output ``i`` samples source coordinate
        ``(i + 0.5) * scale - 0.5`` from its four nearest taps; taps falling outside the
        image clamp to the border (cv2's replicate behavior), accumulating their weight.
        The four kernel weights sum to one, so no renormalization is needed.
        """
        a = -0.75
        scale = src / dst
        out_idx = torch.arange(dst, dtype=torch.float32, device=device)
        center = (out_idx + 0.5) * scale - 0.5
        base = torch.floor(center)
        frac = center - base

        def kernel(t: Tensor) -> Tensor:
            t = t.abs()
            inner = (a + 2) * t**3 - (a + 3) * t**2 + 1
            outer = a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
            return torch.where(
                t <= 1, inner, torch.where(t < 2, outer, torch.zeros_like(t))
            )

        weights = torch.zeros(dst, src, dtype=torch.float32, device=device)
        rows = torch.arange(dst, device=device)
        for offset in (-1, 0, 1, 2):
            taps = (base + offset).long().clamp(min=0, max=src - 1)
            weights[rows, taps] += kernel(frac - offset)
        return weights

    # -----------------------------
    # Make divisible by PATCH_SIZE
    # -----------------------------
    def _make_divisible_by_crop(self, img: Tensor, patch: int) -> Tensor:
        """
        Floor each dimension to the nearest multiple of PATCH_SIZE via center crop.
        Example: 504x377 -> 504x364
        """
        h, w = img.shape[-2:]
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = tv_functional.crop(img, top=top, left=left, height=new_h, width=new_w)
        return img

    def _make_divisible_by_resize(self, img: Tensor, patch: int) -> Tensor:
        """
        Round each dimension to nearest multiple of PATCH_SIZE via small resize.
        """
        h, w = img.shape[-2:]

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch))
        new_h = max(1, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        return self._resize_to(img, new_h, new_w)
