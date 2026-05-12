import numpy as np
from PIL import Image, ImageFilter
from typing import Optional
from skimage.segmentation import slic
from skimage.color import rgb2lab, deltaE_ciede2000
from scipy.ndimage import uniform_filter
import os

from .config import AIConfig, default_config

def apply_slic(
    image: Image.Image,
    color_palette: np.ndarray,
    n_segments: int=None,
    compactness: int=None,
    config: AIConfig=default_config,
    save_steps: bool=False,
    steps_dir: str="test_output/slic_steps"
) -> Image.Image:

    if n_segments is None:
        n_segments = config.SLIC_N_SEGMENTS
    if compactness is None:
        compactness = config.SLIC_COMPACTNESS

    if save_steps:
        os.makedirs(steps_dir, exist_ok=True)

    img_array = np.array(image.convert("RGB"))

    if save_steps:
        image.save(os.path.join(steps_dir, "00_input_before_slic.png"))

    segment = slic(
        img_array,
        n_segments = n_segments,
        compactness = compactness,
        start_label=0
    )

    if save_steps:
        total_segments = segment.max() + 1
        seg_visual = (segment.astype(np.float64) / total_segments * 255).astype(np.uint8)
        Image.fromarray(seg_visual).save(os.path.join(steps_dir, "01_slic_segments_map.png"))
        print(f"[SLIC] Total segments: {total_segments}")

    palette_rgb = color_palette.reshape(1, -1, 3).astype(np.float64)/255.0
    lab_palette = rgb2lab(palette_rgb).reshape(-1,3)

    output = np.zeros_like(img_array)

    total_segments = segment.max() + 1
    save_interval = max(1, total_segments // 10) if save_steps else 0
    step_count = 0

    for segment_id in range(total_segments):

        mask = segment == segment_id

        if not np.any(mask):
            continue

        segment_pixels = img_array[mask]
        mean_rgb = segment_pixels.mean(axis=0)

        mean_lab = rgb2lab(
            mean_rgb.reshape(1,1,3).astype(np.float64)/255.0
        ).reshape(3)

        min_dist = float("inf")
        best_color_idx = 0

        for idx, palette_color_lab in enumerate(lab_palette):
            dist = deltaE_ciede2000(
                mean_lab.reshape(1,1,3),
                palette_color_lab.reshape(1,1,3)
            ).mean()

            if dist < min_dist:
                min_dist = dist
                best_color_idx = idx

        output[mask] = color_palette[best_color_idx]

        if save_steps and save_interval > 0 and (segment_id + 1) % save_interval == 0:
            step_count += 1
            pct = int((segment_id + 1) / total_segments * 100)
            snapshot = Image.fromarray(output)
            snapshot.save(os.path.join(steps_dir, f"02_step_{step_count:02d}_{pct}pct.png"))
            print(f"[SLIC] Step {step_count}: {segment_id + 1}/{total_segments} segments filled ({pct}%)")

    if save_steps:
        Image.fromarray(output).save(os.path.join(steps_dir, "03_final_output.png"))
        print(f"[SLIC] Done! {step_count + 1} step images saved to {steps_dir}/")

    return Image.fromarray(output)

def denoise_image(image: Image.Image, strength: int=10) -> Image.Image:
    return image.filter(ImageFilter.MedianFilter(size=3))

def postprocess_pattern(
    raw_pattern: Image.Image,
    color_palette: np.ndarray,
    apply_segmentation: bool = False,
    apply_denoise: bool = True,
    save_steps: bool = False,
    config: AIConfig = default_config
) -> Image.Image:
    result = raw_pattern

    if apply_denoise:
        result = denoise_image(result)

    if apply_segmentation:
        result = apply_slic(
            result,
            color_palette,
            config=config,
            save_steps=save_steps
        )

    return result
