
import io
import httpx
from PIL import Image
from typing import List, Optional

from ai_model.config import AIConfig, default_config
from ai_model.preprocessing import preprocess_images
from ai_model.pipeline import CamouflagePipeline
from ai_model.postprocessing import postprocess_pattern


class PatternService:
    """
    Manages the AI pipeline lifecycle and generation.

    The model is loaded once at startup and reused for every request.
    """

    def __init__(self):
        self.pipeline: Optional[CamouflagePipeline] = None
        self.config: AIConfig = default_config
        self._is_ready: bool = False

    def startup(self, config: AIConfig = None):
        """
        Load the AI model into memory. Called once at server startup.
        """
        if config:
            self.config = config

        try:
            print("[PatternService] Loading AI pipeline...")
            self.pipeline = CamouflagePipeline.load(self.config)
            self._is_ready = True
            print("[PatternService] AI pipeline ready!")
        except Exception as e:
            print(f"[PatternService] Failed to load AI pipeline: {e}")
            print("[PatternService] Pattern generation will be unavailable.")
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Check if the AI model is loaded and ready to generate."""
        return self._is_ready and self.pipeline is not None

    async def _download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download an image from a Supabase public URL → PIL Image.
        
        Async because network I/O shouldn't block the server.
        httpx is the async-compatible alternative to requests.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=120.0)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                return image
        except Exception as e:
            print(f"[PatternService] Failed to download image: {url} — {e}")
            return None

    async def generate_pattern(
        self,
        image_urls: List[str],
        apply_segmentation: bool = False
    ) -> Optional[bytes]:
        """
        Run the full AI pipeline and return the pattern as raw bytes.

        RETURNS:
            bytes — PNG image data ready for upload_pattern()
            None — if generation failed

        The router receives these bytes and passes them to:
            storage.upload_pattern(file_content=pattern_bytes, ...)
        """
        if not self.is_ready:
            print("[PatternService] AI pipeline not loaded")
            return None

        # ── Step 1: Download base images from Supabase ──
        print(f"[PatternService] Downloading {len(image_urls)} base images...")
        images = []
        for url in image_urls:
            img = await self._download_image(url)
            if img is not None:
                images.append(img)

        if not images:
            print("[PatternService] No images could be downloaded")
            return None

        print(f"[PatternService] Downloaded {len(images)} images")

        # ── Step 2: Preprocess (color extraction + block shuffling) ──
        print("[PatternService] Preprocessing...")
        preprocessed = preprocess_images(images, config=self.config)
        composite_image = preprocessed["composite_image"]
        color_palette = preprocessed["color_palette"]

        print(f"[PatternService] Extracted {len(color_palette)} dominant colors")

        # ── Step 3: Generate via SD3 ──
        print("[PatternService] Running SD3 generation...")
        raw_pattern = self.pipeline.generate(composite_image)

        # ── Step 4: Postprocess (optional SLIC segmentation) ──
        print("[PatternService] Postprocessing...")
        final_pattern = postprocess_pattern(
            raw_pattern=raw_pattern,
            color_palette=color_palette,
            apply_segmentation=apply_segmentation,
            config=self.config
        )

        # ── Step 5: Convert PIL Image → bytes ──
        # Save as PNG for lossless quality
        # JPEG would add compression artifacts to the pattern
        img_buffer = io.BytesIO()
        final_pattern.save(img_buffer, format="PNG")
        pattern_bytes = img_buffer.getvalue()

        print(f"[PatternService] Pattern generated ({len(pattern_bytes)} bytes)")
        return pattern_bytes


# Singleton instance — shared across the entire app
pattern_service = PatternService()