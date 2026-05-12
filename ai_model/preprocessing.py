import numpy as np
from PIL import Image
from typing import List,Tuple
from sklearn.cluster import KMeans

from .config import AIConfig, default_config

def extract_dominant_colors(images: List[Image.Image], num_colors: int=None, config: AIConfig = default_config) -> np.ndarray:
    if num_colors is None:
        num_colors = config.NUM_COLORS

    all_pixels = []

    for img in images:
        img_rgb = img.convert("RGB")

        img_small_resize = img_rgb.resize((100,100))

        pixels = np.array(img_small_resize)

        pxl_flat = pixels.reshape(-1,3)

        all_pixels.append(pxl_flat)

    all_pixels = np.vstack(all_pixels)

    k_means = KMeans(
        n_clusters= num_colors,
        n_init=10,
        random_state=42
    ).fit(all_pixels)

    dominant_col = k_means.cluster_centers_.astype(np.uint8)

    return dominant_col


def shuffle_image(
    images: List[Image.Image],
    block_size: int = None,
    output_size: Tuple[int, int] = None,
    config: AIConfig = default_config
) -> Image.Image:

    """
    Divide image(s) intu chukz. Combine the chunks than shuffle it. The image is destroyed but
    Every blocks still has it real colors
    """

    if block_size is None:
        block_size = config.SHUFFLE_BLOCK_SIZE
    if output_size is None:
        output_size = (config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)

    output_w, output_h = output_size

    all_blocks = []

    for image in images:
        img_rgb = image.convert("RGB")

        w, h = img_rgb.size

        cols = w // block_size
        rows = h // block_size

        for row in range(rows):
            for col in range(cols):
                left = col * block_size
                top = row * block_size
                right = left + block_size
                bottom = top + block_size

                block = img_rgb.crop((left,top,right,bottom))
                all_blocks.append(block)


    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_blocks)

    grid_cols = output_w // block_size
    grid_row = output_h // block_size
    total = grid_cols * grid_row


    if len(all_blocks) < total:
        repeat = (total + len(all_blocks))+1
        all_blocks = (all_blocks*repeat)[:total]
    else:
        all_blocks = all_blocks[:total]

    output_image = Image.new("RGB", (output_w, output_h))

    for idx, block in enumerate(all_blocks):
        col = idx % grid_cols
        row = idx // grid_cols
        x = col * block_size
        y = row * block_size
        output_image.paste(block, (x, y))

    return output_image


def preprocess_images(
    images: List[Image.Image],
    config: AIConfig = default_config
) -> dict:
    """
    Main preprocessing function that runs the full pipeline.

    This is the function the service layer will call.
    It returns everything the pipeline needs:
        - The shuffled composite image (input for SD3)
        - The dominant color palette (for post-processing)

    PARAMETERS:
        images: List of PIL Image objects from user upload
        config: AI configuration settings

    RETURNS:
        Dictionary with:
            "composite_image": PIL Image — the shuffled grid
            "color_palette": numpy array — dominant colors (N, 3)
    """
    color_palette = extract_dominant_colors(images, config=config)

    composite = shuffle_image(images, config=config)

    return {
        "composite_image": composite,
        "color_palette": color_palette
    }
