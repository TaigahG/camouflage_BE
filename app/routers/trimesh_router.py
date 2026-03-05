import io
import os
import tempfile

import trimesh
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

router = APIRouter()


@router.post(
    "/apply-pattern",
    summary="Apply a pattern texture to an uploaded .glb model and download the result",
    response_description="Textured GLB file",
)
async def apply_pattern(
    model: UploadFile = File(..., description="GLB model file (.glb)"),
    pattern: UploadFile = File(..., description="Pattern image (JPEG / PNG / WebP)"),
):
    """
    Upload a .glb model and a pattern image.
    Returns the model with the pattern texture applied to every UV-mapped mesh
    as a downloadable .glb file.
    """
    # Validate model file
    if not model.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Model must be a .glb file.")

    # Validate image type
    allowed_image_types = {"image/jpeg", "image/png", "image/webp"}
    if pattern.content_type not in allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{pattern.content_type}'. Use JPEG, PNG, or WebP.",
        )

    model_bytes = await model.read()
    image_bytes = await pattern.read()

    # Decode texture image
    try:
        texture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open pattern image: {exc}")

    with tempfile.TemporaryDirectory() as td:
        # Write the uploaded GLB to a temp file so trimesh can load it
        glb_in = os.path.join(td, "input.glb")
        with open(glb_in, "wb") as f:
            f.write(model_bytes)

        # Load the model
        try:
            scene = trimesh.load(glb_in)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cannot load model: {exc}")

        # Apply texture to every mesh that has UV coordinates
        geometries = scene.geometry if hasattr(scene, "geometry") else {"mesh": scene}
        applied = 0
        for name, mesh in geometries.items():
            if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
                continue
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(image=texture)
            applied += 1

        if applied == 0:
            raise HTTPException(
                status_code=422,
                detail="No UV-mapped meshes found in the model; texture cannot be applied.",
            )

        # Export textured model to GLB
        glb_out = os.path.join(td, "textured_model.glb")
        try:
            scene.export(glb_out)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

        with open(glb_out, "rb") as f:
            glb_bytes = f.read()

    return StreamingResponse(
        io.BytesIO(glb_bytes),
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=textured_model.glb"},
    )
