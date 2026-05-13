import torch
from PIL import Image
from typing import Optional

from .config import AIConfig, default_config

class CamouflagePipeline:

    def __init__(self, pipe, config: AIConfig):
        self.pipe = pipe
        self.config = config
        # Lazily constructed inpaint pipeline that shares weights with `pipe`
        self._inpaint_pipe = None


    @classmethod
    def load(cls, config: AIConfig = default_config) -> "CamouflagePipeline":
        from diffusers import StableDiffusion3Img2ImgPipeline

        if config.DEVICE == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.float16

        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            config.MODEL,
            torch_dtype=dtype,
            text_encoder_3=None,
            tokenizer_3=None
        )

        if config.LORA_WEIGHTS_PATH:
            print(f"Loading LoRA weights: {config.LORA_WEIGHTS_PATH}")
            pipe.load_lora_weights(config.LORA_WEIGHTS_PATH)

        pipe = pipe.to(config.DEVICE)

        # Attention slicing saves memory but kills speed on high-VRAM GPUs
        # RTX 3070 has 8GB, so we don't need it
        # pipe.enable_attention_slicing()  # DISABLED for speed

        print(f"Model loaded successfully on {config.DEVICE}")

        return cls(pipe=pipe, config=config)

    def _get_inpaint_pipe(self):
        """
        Build the SD3 inpaint pipeline on first use, reusing the loaded weights
        from the img2img pipeline (shares transformer/VAE/text encoders).
        """
        if self._inpaint_pipe is not None:
            return self._inpaint_pipe

        from diffusers import StableDiffusion3InpaintPipeline

        print("[AI] Constructing SD3 inpaint pipeline (sharing weights)...")
        self._inpaint_pipe = StableDiffusion3InpaintPipeline.from_pipe(self.pipe)
        return self._inpaint_pipe

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        """
        Run SD3 inpainting: regenerate ONLY the white area of `mask` to match
        `prompt`. The rest of `image` is preserved. Defaults are read from
        `AIConfig.INPAINT_*` if not overridden.
        """
        if strength is None:
            strength = self.config.INPAINT_STRENGTH
        if num_steps is None:
            num_steps = self.config.INPAINT_NUM_INFERENCE_STEPS
        if guidance_scale is None:
            guidance_scale = self.config.INPAINT_GUIDANCE_SCALE

        inpaint_pipe = self._get_inpaint_pipe()

        # Inpaint at INPAINT_WIDTH/HEIGHT (typically smaller than the img2img
        # OUTPUT_*) — the result is resized back to the input's native size.
        target_w = self.config.INPAINT_WIDTH
        target_h = self.config.INPAINT_HEIGHT

        rgb = image.convert("RGB")
        mask_l = mask.convert("L")

        original_size = rgb.size  # (W, H) so we can restore later
        rgb_resized = rgb.resize((target_w, target_h), Image.LANCZOS)
        mask_resized = mask_l.resize((target_w, target_h), Image.NEAREST)

        print(f"[AI] Inpainting clothes region...")
        print(f"[AI]   Prompt: {prompt}")
        print(f"[AI]   Strength: {strength}, Steps: {num_steps}, Guidance: {guidance_scale}")

        with torch.no_grad():
            result = inpaint_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=rgb_resized,
                mask_image=mask_resized,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=target_h,
                width=target_w,
            )

        generated = result.images[0]
        # Restore original resolution so the photo matches the user's input
        if generated.size != original_size:
            generated = generated.resize(original_size, Image.LANCZOS)

        print(f"[AI] Inpainting done: {generated.size}")
        return generated

    def generate(
        self,
        input_image: Image.Image,
        prompt: str = "",
        strength: Optional[float] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:

        if strength is None:
            strength = self.config.STRENGTH
        if num_steps is None:
            num_steps = self.config.NUM_INFERENEC_STEPS
        if guidance_scale is None:
            guidance_scale = self.config.GUIDANCE_SCALE

        target_size = (self.config.OUTPUT_HEIGHT, self.config.OUTPUT_WIDTH)
        if input_image.size != target_size:
            input_image = input_image.resize(target_size, Image.LANCZOS)

        input_image = input_image.convert("RGB")

        print(f"[AI] Generating pattern...")
        print(f"[AI]   Strength: {strength}")
        print(f"[AI]   Steps: {num_steps} (effective: {int(num_steps * strength)})")
        print(f"[AI]   Guidance: {guidance_scale}")

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )

        generated_image = result.images[0]

        print(f"[AI] Pattern generated: {generated_image.size}")
        return generated_image

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.pipe is not None
