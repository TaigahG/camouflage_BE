"""
test_model.py

PURPOSE:
    Test the AI pipeline in isolation — no FastAPI, no Supabase, no database.
    Just: local images → preprocessing → SD3 → postprocessing → saved file.

    Run this FIRST before connecting to the app to verify:
    1. PyTorch + MPS works on your M3
    2. SD3 downloads and loads successfully
    3. Preprocessing produces a valid shuffled composite
    4. Generation runs without errors
    5. The output looks like something resembling camouflage

HOW TO RUN:
    1. Put 1-5 environment photos in a folder (e.g., test_images/)
    2. Run: python test_model.py
    3. Check the output files in test_output/

EXPECTED TIMELINE (first run on M3 18GB):
    - Model download: 5-15 minutes (one time only, ~4-5GB)
    - Model loading: 10-20 seconds
    - Preprocessing: 1-2 seconds
    - Generation: 30-90 seconds
    - Total first run: ~20 minutes (mostly download)
    - Subsequent runs: ~1-2 minutes
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Folder containing your test environment photos
# Put some photos here before running!
INPUT_FOLDER = "test_images"

# Where results will be saved
OUTPUT_FOLDER = "test_output"

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def check_environment():
    """
    Verify everything is installed and your M3 GPU is detected.
    Run this before trying to load the model.
    """
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    # Check Python version
    print(f"\nPython: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  MPS available:  {torch.backends.mps.is_available()}")

        if torch.backends.mps.is_available():
            print(f"  ✅ MPS (Apple Metal) detected — your M3 GPU will be used")
        elif torch.cuda.is_available():
            print(f"  ✅ CUDA detected — NVIDIA GPU will be used")
        else:
            print(f"  ⚠️  No GPU detected — will use CPU (very slow)")
    except ImportError:
        print("  ❌ PyTorch not installed! Run: pip install torch")
        return False

    # Check diffusers
    try:
        import diffusers
        print(f"Diffusers: {diffusers.__version__}")
    except ImportError:
        print("  ❌ Diffusers not installed! Run: pip install diffusers")
        return False

    # Check transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("  ❌ Transformers not installed! Run: pip install transformers")
        return False

    # Check HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"HF_TOKEN: set ({hf_token[:8]}...)")
    else:
        print("  ⚠️  HF_TOKEN not set — you need this to download SD3")
        print("     Set it in .env or run: export HF_TOKEN=hf_your_token")
        return False

    # Check for test images
    image_files = _get_image_files()
    if image_files:
        print(f"\nTest images found: {len(image_files)}")
        for f in image_files:
            print(f"  - {f.name}")
    else:
        print(f"\n⚠️  No images in {INPUT_FOLDER}/")
        print(f"   Put some environment photos there first!")
        return False

    print("\n" + "=" * 60)
    print("✅ All checks passed — ready to test")
    print("=" * 60)
    return True


def _get_image_files() -> list:
    """Find all image files in the input folder."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    folder = Path(INPUT_FOLDER)
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.suffix.lower() in extensions]


def test_preprocessing():
    """
    Test ONLY the preprocessing step (no model loading needed).
    Fast way to verify color extraction and shuffling work.
    """
    print("\n" + "=" * 60)
    print("TEST: Preprocessing Only")
    print("=" * 60)

    from ai_model.config import default_config
    from ai_model.preprocessing import preprocess_images

    # Load test images
    image_files = _get_image_files()
    images = [Image.open(f) for f in image_files]
    print(f"Loaded {len(images)} images")

    # Run preprocessing
    start = time.time()
    result = preprocess_images(images, config=default_config)
    elapsed = time.time() - start

    composite = result["composite_image"]
    palette = result["color_palette"]

    print(f"Preprocessing took: {elapsed:.2f}s")
    print(f"Composite size: {composite.size}")
    print(f"Dominant colors ({len(palette)}):")
    for i, color in enumerate(palette):
        print(f"  Color {i+1}: RGB({color[0]}, {color[1]}, {color[2]})")

    # Save the composite so you can inspect it
    composite_path = os.path.join(OUTPUT_FOLDER, "01_shuffled_composite.png")
    composite.save(composite_path)
    print(f"\nSaved: {composite_path}")

    # Save a color palette visualization
    _save_palette_image(palette, os.path.join(OUTPUT_FOLDER, "02_color_palette.png"))

    return result


def test_full_pipeline():
    """
    Test the complete pipeline: preprocessing → SD3 generation → postprocessing.
    This will load the model and run actual diffusion.
    """
    print("\n" + "=" * 60)
    print("TEST: Full Pipeline (with SD3)")
    print("=" * 60)

    from ai_model.config import default_config
    from ai_model.preprocessing import preprocess_images
    from ai_model.pipeline import CamouflagePipeline
    from ai_model.postprocessing import postprocess_pattern

    # Load test images
    image_files = _get_image_files()
    images = [Image.open(f) for f in image_files]
    print(f"Loaded {len(images)} images")

    # ── Preprocessing ──
    print("\n--- Preprocessing ---")
    start = time.time()
    preprocessed = preprocess_images(images, config=default_config)
    print(f"Preprocessing: {time.time() - start:.2f}s")

    composite = preprocessed["composite_image"]
    palette = preprocessed["color_palette"]

    # Save composite
    composite.save(os.path.join(OUTPUT_FOLDER, "01_shuffled_composite.png"))

    # ── Load Model ──
    print("\n--- Loading SD3 Model ---")
    print("(First run downloads ~4-5GB — grab a coffee)")
    start = time.time()
    pipeline = CamouflagePipeline.load(default_config)
    print(f"Model loaded: {time.time() - start:.2f}s")

    # ── Generate (no segmentation — raw SD3 output) ──
    print("\n--- Generating Pattern (raw, no SLIC) ---")
    start = time.time()
    raw_pattern = pipeline.generate(composite)
    gen_time = time.time() - start
    print(f"Generation: {gen_time:.2f}s")

    # Save raw output
    raw_path = os.path.join(OUTPUT_FOLDER, "03_pattern_raw.png")
    raw_pattern.save(raw_path)
    print(f"Saved: {raw_path}")

    # ── Postprocess WITHOUT segmentation (like Gen1 in the thesis) ──
    print("\n--- Postprocessing (no SLIC) ---")
    start = time.time()
    pattern_no_slic = postprocess_pattern(
        raw_pattern=raw_pattern,
        color_palette=palette,
        apply_segmentation=False,
        config=default_config
    )
    print(f"Postprocess (no SLIC): {time.time() - start:.2f}s")

    no_slic_path = os.path.join(OUTPUT_FOLDER, "04_pattern_no_slic.png")
    pattern_no_slic.save(no_slic_path)
    print(f"Saved: {no_slic_path}")

    # ── Postprocess WITH segmentation (like Seg1 in the thesis) ──
    print("\n--- Postprocessing (with SLIC) ---")
    start = time.time()
    pattern_with_slic = postprocess_pattern(
        raw_pattern=raw_pattern,
        color_palette=palette,
        apply_segmentation=True,
        config=default_config,
        save_steps=True
    )
    print(f"Postprocess (with SLIC): {time.time() - start:.2f}s")

    slic_path = os.path.join(OUTPUT_FOLDER, "05_pattern_with_slic.png")
    pattern_with_slic.save(slic_path)
    print(f"Saved: {slic_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Device used: {default_config.DEVICE}")
    print(f"\nOutput files in {OUTPUT_FOLDER}/:")
    print(f"  01_shuffled_composite.png  ← What SD3 received as input")
    print(f"  02_color_palette.png       ← Extracted environment colors")
    print(f"  03_pattern_raw.png         ← Raw SD3 output")
    print(f"  04_pattern_no_slic.png     ← After denoising (no SLIC)")
    print(f"  05_pattern_with_slic.png   ← After SLIC segmentation")
    print(f"\nCompare 04 vs 05 to see the effect of SLIC segmentation.")
    print(f"The thesis found 04 (no SLIC) often works better for digital camo.")
    print("=" * 60)


def _save_palette_image(palette, path, swatch_size=100):
    """Create a simple visualization of the color palette."""
    n_colors = len(palette)
    img = Image.new("RGB", (swatch_size * n_colors, swatch_size))

    for i, color in enumerate(palette):
        for x in range(swatch_size):
            for y in range(swatch_size):
                img.putpixel((i * swatch_size + x, y), tuple(color))

    img.save(path)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# MAIN — Run the tests
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Login to HuggingFace (needed to download SD3)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    print("\n🎯 Camouflage AI — Model Test")
    print("=" * 60)

    # Step 1: Check everything is installed
    if not check_environment():
        print("\n❌ Fix the issues above, then try again.")
        sys.exit(1)

    # Step 2: Test preprocessing (fast, no model needed)
    print("\n\nWould you like to:")
    print("  1. Test preprocessing only (fast, no GPU needed)")
    print("  2. Test full pipeline with SD3 (slow first time, needs GPU)")
    print("  3. Run both")

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        test_preprocessing()
    elif choice == "2":
        test_full_pipeline()
    elif choice == "3":
        test_preprocessing()
        test_full_pipeline()
    else:
        print("Invalid choice. Running preprocessing only.")
        test_preprocessing()

    print("\n✅ Done!")