"""
Retexture service — applies a camouflage style to the upper-body clothing
in a user-supplied photo using clothes segmentation + SD3 inpainting.

Flow:
    1. Download the collection's base images (for the color palette).
    2. Extract dominant colors via KMeans.
    3. Segment the user photo to get an upper-body clothing mask.
    4. Build a color-aware prompt describing the camouflage style.
    5. Run SD3 inpainting to retexture only the masked region.
"""
import io
import httpx
import numpy as np
from PIL import Image
from typing import List, Optional

from ai_model.config import AIConfig, default_config
from ai_model.preprocessing import extract_dominant_colors
from ai_model.segmentation import ClothingSegmenter

# Import the shared CamouflagePipeline instance through PatternService
# so we only ever load SD3 once.
from .pattern_service import pattern_service


def _rgb_to_color_name(rgb: np.ndarray) -> str:
    """Map an (R, G, B) tuple to a coarse color word for prompt building."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    mx = max(r, g, b)
    mn = min(r, g, b)

    # Grayscale-ish
    if mx - mn < 25:
        if mx < 60:
            return "black"
        if mx < 130:
            return "dark gray"
        if mx < 200:
            return "gray"
        return "white"

    # Earth tones — heuristics tuned for camouflage palettes
    if r > 100 and g > 80 and b < 80 and abs(r - g) < 60:
        return "olive green" if g > r else "brown"
    if g >= r and g >= b:
        return "dark green" if g < 120 else "green"
    if r >= g and r >= b:
        return "tan" if g > 100 else "rust"
    if b >= r and b >= g:
        return "navy" if b < 150 else "blue"
    return "khaki"


def _palette_words(colors: np.ndarray, max_words: int = 4) -> str:
    color_words = []
    seen = set()
    for rgb in colors:
        word = _rgb_to_color_name(rgb)
        if word not in seen:
            seen.add(word)
            color_words.append(word)
        if len(color_words) >= max_words:
            break
    return ", ".join(color_words) if color_words else "earth-tone"


def _build_prompt(colors: np.ndarray) -> str:
    """Turn a palette into a natural prompt describing the camo style."""
    palette = _palette_words(colors)
    return (
        f"a shirt with a {palette} camouflage pattern, "
        "military fabric texture, blotchy organic camo print, "
        "photorealistic, detailed, natural lighting, fabric folds"
    )


def _build_outfit_prompt(colors: np.ndarray, outfit_type: str) -> str:
    """Outfit-aware prompt for clothing-only images."""
    palette = _palette_words(colors)
    return (
        f"a {outfit_type} with a {palette} camouflage pattern, "
        "military fabric texture, blotchy organic camo print, "
        "photorealistic, detailed fabric weave, natural lighting, fabric folds"
    )


def _outfit_mask(image: Image.Image) -> Image.Image:
    """
    Binary mask for a clothing-only image: white = clothing, black = background.
    Uses the alpha channel when present; otherwise thresholds against a near-white
    background. Slightly dilated so inpainting blends past the silhouette edge.
    """
    from PIL import ImageFilter

    if image.mode == "RGBA":
        alpha = np.array(image.split()[-1])
        mask_arr = (alpha > 50).astype(np.uint8) * 255
    else:
        arr = np.array(image.convert("RGB"))
        # Background = pixels that are near white in all 3 channels
        is_bg = arr.min(axis=2) > 230
        mask_arr = ((~is_bg).astype(np.uint8)) * 255

    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(7))
    return mask_img


_NEGATIVE_PROMPT = (
    "smooth, plain, solid color, low quality, blurry, distorted, "
    "extra limbs, deformed, text, watermark, logo"
)


class RetextureService:
    """
    Lifecycle is tied to PatternService — it must be loaded first.
    Segmentation model is loaded lazily on first request.
    """

    def __init__(self):
        self.config: AIConfig = default_config
        self._segmenter: Optional[ClothingSegmenter] = None

    @property
    def is_ready(self) -> bool:
        return pattern_service.is_ready

    def _get_segmenter(self) -> ClothingSegmenter:
        if self._segmenter is None:
            self._segmenter = ClothingSegmenter.load(device=self.config.DEVICE)
        return self._segmenter

    def preload(self) -> None:
        """
        Eagerly load the segmenter and construct the SD3 inpaint pipeline so
        the first /retexture-clothes request doesn't pay for downloads or
        pipeline assembly. Safe to call once SD3 (PatternService) is ready.
        """
        if not pattern_service.is_ready:
            print("[RetextureService] SD3 not ready, skipping preload.")
            return

        print("[RetextureService] Preloading inpaint pipeline...")
        pattern_service.pipeline._get_inpaint_pipe()

        print("[RetextureService] Preloading clothing segmenter...")
        self._get_segmenter()

        print("[RetextureService] Preload complete.")

    async def _download_image(self, url: str) -> Optional[Image.Image]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"[RetextureService] Failed to download {url}: {e}")
            return None

    async def retexture_upper_body(
        self,
        photo_bytes: bytes,
        base_image_urls: List[str],
    ) -> Optional[bytes]:
        """
        Return PNG bytes of the photo with the upper-body clothing retextured.
        Returns None on failure.
        """
        if not self.is_ready:
            print("[RetextureService] SD3 pipeline is not ready.")
            return None

        # 1) Load the user photo
        try:
            photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        except Exception as e:
            print(f"[RetextureService] Bad photo bytes: {e}")
            return None

        # 2) Download base images and extract palette
        palette_images = []
        for url in base_image_urls:
            img = await self._download_image(url)
            if img is not None:
                palette_images.append(img)

        if not palette_images:
            print("[RetextureService] No base images available; using neutral prompt")
            prompt = "a shirt with an earth-tone camouflage pattern, military fabric, photorealistic"
        else:
            palette = extract_dominant_colors(palette_images, config=self.config)
            prompt = _build_prompt(palette)

        # 3) Segment upper-body clothing
        segmenter = self._get_segmenter()
        mask = segmenter.segment_upper_body(photo)

        # Sanity check: did we actually find clothing?
        mask_array = np.array(mask)
        if mask_array.max() == 0:
            print("[RetextureService] No upper-body clothing detected in photo.")
            return None

        # 4) Inpaint via SD3 (shares weights with the existing img2img pipe)
        pipeline = pattern_service.pipeline
        result = pipeline.inpaint(
            image=photo,
            mask=mask,
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
        )

        # 5) PNG bytes for the response
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    async def retexture_outfit(
        self,
        outfit_bytes: bytes,
        base_image_urls: List[str],
        outfit_type: str = "shirt",
    ) -> Optional[bytes]:
        """
        Apply a collection's camo style to a clothing-only image (no person).
        The mask is derived from the alpha channel or a near-white background,
        not from SegFormer. Background pixels are preserved transparent.
        Returns PNG bytes or None on failure.
        """
        if not self.is_ready:
            print("[RetextureService] SD3 pipeline is not ready.")
            return None

        try:
            outfit_img = Image.open(io.BytesIO(outfit_bytes))
        except Exception as e:
            print(f"[RetextureService] Bad outfit bytes: {e}")
            return None

        # Palette → prompt
        palette_images = []
        for url in base_image_urls:
            img = await self._download_image(url)
            if img is not None:
                palette_images.append(img)

        if not palette_images:
            prompt = (
                f"a {outfit_type} with an earth-tone camouflage pattern, "
                "military fabric, photorealistic"
            )
        else:
            palette = extract_dominant_colors(palette_images, config=self.config)
            prompt = _build_outfit_prompt(palette, outfit_type)

        # Mask from alpha/background
        mask = _outfit_mask(outfit_img)
        if np.array(mask).max() == 0:
            print("[RetextureService] No clothing region detected in outfit image.")
            return None

        # SD3 inpaint wants RGB. Composite onto white so the unmasked region
        # stays clean (the inpaint will only touch the masked area).
        if outfit_img.mode == "RGBA":
            white_bg = Image.new("RGB", outfit_img.size, (255, 255, 255))
            white_bg.paste(outfit_img, mask=outfit_img.split()[-1])
            rgb_outfit = white_bg
        else:
            rgb_outfit = outfit_img.convert("RGB")

        pipeline = pattern_service.pipeline
        result = pipeline.inpaint(
            image=rgb_outfit,
            mask=mask,
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
        )

        # Restore transparency outside the clothing silhouette so the PNG
        # composites cleanly back into the UI.
        result_rgba = result.convert("RGBA")
        alpha = mask if mask.size == result.size else mask.resize(result.size, Image.NEAREST)
        result_rgba.putalpha(alpha)

        buf = io.BytesIO()
        result_rgba.save(buf, format="PNG")
        return buf.getvalue()


retexture_service = RetextureService()
