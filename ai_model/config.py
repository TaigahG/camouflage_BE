import os
from dotenv import load_dotenv

load_dotenv()

class AIConfig():
    MODEL: str = "stabilityai/stable-diffusion-3-medium-diffusers"

    LORA_WEIGHTS_PATH: str = os.getenv("LORA_WEIGHTS_PATH", None)

    DTYPE: str = "float16"

    # GENERATION SETTINGS

    #Strength power (0-1)
    #Reimagine vs preserves. Higher value ignores colors/textures. Lower values looks to shuffled
    STRENGTH: float = 0.70

    #Denoising steps
    #More steps higher quality but slower. To fine an effective steps is: N_I_S * STRENGTH. Q: Why 50?
    NUM_INFERENEC_STEPS: int = 30

    #Guidance scale
    #To know how much the model follows the text prompt.
    GUIDANCE_SCALE: float = 0.0

    # INPAINTING SETTINGS (for /retexture-* endpoints)
    # Inpainting needs *different* defaults than img2img:
    #  - higher strength so the masked region is fully regenerated
    #  - non-zero guidance so the camo text prompt actually influences the result
    INPAINT_STRENGTH: float = 0.70
    # 15–20 steps is the sweet spot for SD3 inpaint on camo (which is high-frequency
    # texture; further denoising barely helps). 15 ≈ 2× faster than 30.
    INPAINT_NUM_INFERENCE_STEPS: int = 15
    INPAINT_GUIDANCE_SCALE: float = 7.0
    # Inpaint at a smaller resolution than the img2img generator. The masked
    # region is resized in/out anyway, so 512² gives ~2× speedup over 768²
    # with no visible loss for camo texture.
    INPAINT_WIDTH: int = 512
    INPAINT_HEIGHT: int = 512

    #IMAGE SETTINGS
    
    #Output dimensions
    OUTPUT_WIDTH: int = 768
    OUTPUT_HEIGHT: int = 768


    #PREPROCESS SETTINGS

    #Block size image shuffling. To destroy spatial relationship while keeping the texture/color
    #   - Small blocks (4-8): Very noisy, fine color mixing
    #   - Medium blocks (16-32): Good balance (thesis used 16 for latent, 100 for pixel)
    #   - Large blocks (100+): Preserves too much spatial structure
    SHUFFLE_BLOCK_SIZE: int = 32

    #Shuffle in latent or pixel space. Latent space shuffling produced smoother transition
    SHUFFLE_IN_LATENT: bool = True

    #Number of color to extract
    NUM_COLORS: int = 6

    
    #POSTPROCESSING SETTINGS

    #SLIC segmentation parameters (it clusters pixels based on their color similarity and proximity)
    # determine superpixels region
    SLIC_N_SEGMENTS: int = 200 
    #higher = more square region
    SLIC_COMPACTNESS: float = 17.0
    # Final color in palette
    SLIC_K_COLORS: int = 6



    # DEVICE SETTINGS
    DEVICE: str = os.getenv("AI_DEVICE", None)

    def __repr__(self):
        return (
            f"AIConfig(\n"
            f"  model={self.MODEL}\n"
            f"  strength={self.STRENGTH}\n"
            f"  steps={self.NUM_INFERENEC_STEPS}\n"
            f"  device={self.DEVICE}\n"
            f")"
        )


def _detect_device() -> str:
    """Auto-detect the best available device: CUDA > MPS > CPU"""
    import torch

    if torch.cuda.is_available():
        print("[Device Detection] CUDA GPU detected and available")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("[Device Detection] Apple MPS (Metal Performance Shaders) detected")
        return "mps"
    else:
        print("[Device Detection] No GPU available, falling back to CPU")
        return "cpu"



# Auto-detect device on module load
_detected_device = _detect_device()

# Create default config and set the device
default_config = AIConfig()
if default_config.DEVICE is None:
    default_config.DEVICE = _detected_device
else:
    print(f"[Config] Using AI_DEVICE from environment: {default_config.DEVICE}")
