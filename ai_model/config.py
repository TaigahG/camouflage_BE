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
    NUM_INFERENEC_STEPS: int = 50

    #Guidance scale
    #To know how much the model follows the text prompt. 
    GUIDANCE_SCALE: float = 0.0

    #IMAGE SETTINGS
    
    #Output dimensions
    OUTPUT_WIDTH: int = 1024
    OUTPUT_HEIGHT: int = 1024


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
    DEVICE: str = os.getenv("AI_DEVICE", "cuda")

def _detect_device() -> str:

    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

    def __repr__(self):
        return (
            f"AIConfig(\n"
            f"  model={self.MODEL_ID}\n"
            f"  strength={self.STRENGTH}\n"
            f"  steps={self.NUM_INFERENCE_STEPS}\n"
            f"  device={self.DEVICE}\n"
            f")"
        )



default_config = AIConfig()


    