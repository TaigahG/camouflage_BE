"""
ai_model/__init__.py

This makes the ai_model/ folder a Python package.
Without this file, Python won't recognize imports like:
    from ai_model.config import default_config

We also expose the main entry point here so the API layer
can do a simple:
    from ai_model import CamouflagePipeline
"""

from .config import AIConfig, default_config

# These will be importable once we create the files:
# from .preprocessing import preprocess_images
# from .pipeline import CamouflagePipeline
# from .postprocessing import postprocess_pattern