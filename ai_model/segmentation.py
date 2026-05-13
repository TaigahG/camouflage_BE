"""
Clothing segmentation module.

Uses `mattmdjaga/segformer_b2_clothes` to segment clothing from a person photo.
We only care about upper-body clothing classes (Upper-clothes=4, Dress=7).
"""
from typing import Optional

import numpy as np
import torch
from PIL import Image


# SegFormer clothes classes — full list for reference:
#   0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes,
#   5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
#   11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
#   16=Bag, 17=Scarf
UPPER_BODY_CLASSES = (4, 7)


class ClothingSegmenter:
    """
    Wraps the SegFormer clothes model. Loaded once and reused.
    """

    MODEL_ID = "mattmdjaga/segformer_b2_clothes"

    def __init__(self, processor, model, device: str):
        self.processor = processor
        self.model = model
        self.device = device

    @classmethod
    def load(cls, device: str = "cuda") -> "ClothingSegmenter":
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

        print(f"[Segmenter] Loading {cls.MODEL_ID} on {device}...")

        processor = SegformerImageProcessor.from_pretrained(cls.MODEL_ID)
        model = AutoModelForSemanticSegmentation.from_pretrained(cls.MODEL_ID)
        model = model.to(device)
        model.eval()

        print("[Segmenter] Model loaded.")
        return cls(processor=processor, model=model, device=device)

    @torch.no_grad()
    def segment_upper_body(
        self,
        image: Image.Image,
        dilate_pixels: int = 6,
    ) -> Image.Image:
        """
        Return a binary mask (mode "L") matching `image.size` where
        white (255) = upper-body clothing, black (0) = everything else.

        `dilate_pixels` slightly expands the mask so the inpainting blends
        cleanly past the segmentation boundary.
        """
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, num_classes, H/4, W/4)

        # Upsample logits to original image size
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=rgb.size[::-1],  # PIL is (W, H), torch needs (H, W)
            mode="bilinear",
            align_corners=False,
        )

        pred = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W)

        mask = np.zeros(pred.shape, dtype=np.uint8)
        for cls_id in UPPER_BODY_CLASSES:
            mask[pred == cls_id] = 255

        if dilate_pixels > 0:
            mask = _dilate(mask, dilate_pixels)

        return Image.fromarray(mask, mode="L")


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Cheap binary dilation via PIL's MaxFilter — avoids pulling in cv2."""
    from PIL import ImageFilter
    img = Image.fromarray(mask, mode="L")
    # MaxFilter kernel must be odd
    k = max(3, radius * 2 + 1)
    img = img.filter(ImageFilter.MaxFilter(k))
    return np.array(img)
