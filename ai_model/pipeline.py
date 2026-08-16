import torch
from PIL import Image
from typing import Optional

from .config import AIConfig, default_config

class CamouflagePipeline:

    def __init__(self, pipe, config: AIConfig):
        self.pipe = pipe
        self.config = config


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


        pipe.enable_attention_slicing() 

        print(f"Model loaded successfully on {config.DEVICE}")

        return cls(pipe=pipe, config=config)

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
