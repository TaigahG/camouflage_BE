import io
import os
import tempfile
from typing import Optional
from urllib.request import urlopen

import trimesh
from fastapi import APIRouter, File, HTTPException, UploadFile, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from sqlalchemy.orm import Session

from .. import crud, schemas
from ..database import get_db
from ..auth import get_current_user
from ..schemas import UserInfo
from ..storage import upload_applied_model
from .. import models

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


@router.post(
    "/apply-pattern-and-save",
    response_model=schemas.AppliedPatternResponse,
    status_code=201,
    summary="Apply pattern to model and save result",
)
async def apply_pattern_and_save(
    model: UploadFile = File(..., description="GLB model file (.glb)"),
    collection_id: int = Form(..., description="Collection ID"),
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Apply a pattern to a 3D model and save the result to storage and database.
    
    Args:
        model: GLB model file
        collection_id: Collection/Pattern ID
        db: Database session
        current_user: Current authenticated user
    
    Returns:
        AppliedPatternResponse with the created applied pattern record
    """
    user_id = current_user.id

    # Validate model file
    if not model.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Model must be a .glb file.")

    # Verify collection exists and belongs to user
    db_collection = crud.get_collections(db, collection_id)
    if not db_collection or db_collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Collection not found or access denied")

    model_bytes = await model.read()

    # Get pattern image from collection
    if not db_collection.pattern_image_url:
        raise HTTPException(status_code=404, detail="Collection has no pattern image")

    # Download pattern image from URL
    try:
        with urlopen(db_collection.pattern_image_url) as response:
            image_bytes = response.read()
        texture = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot download or open pattern image: {exc}")

    with tempfile.TemporaryDirectory() as td:
        # Write the uploaded GLB to a temp file
        glb_in = os.path.join(td, "input.glb")
        with open(glb_in, "wb") as f:
            f.write(model_bytes)

        # Load the model
        try:
            scene = trimesh.load(glb_in)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cannot load model: {exc}")

        # Apply texture to every mesh
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

    # Create applied pattern record first to get the ID
    db_applied = crud.create_applied_pattern(
        db=db,
        user_id=user_id,
        collection_id=collection_id,
        applied_model_url="placeholder",  # Will be updated after upload
    )

    # Upload the textured model to storage
    try:
        model_url = upload_applied_model(
            file_content=glb_bytes,
            filename="textured_model.glb",
            user_id=user_id,
            applied_id=db_applied.applied_id,
            pattern_id=collection_id,
            file_type="model",
        )

        # Update the applied pattern with the actual model URL
        db_applied.applied_model_url = model_url
        db.commit()
        db.refresh(db_applied)

    except Exception as e:
        # Rollback if upload fails
        crud.delete_applied_pattern(db, db_applied.applied_id)
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

    return db_applied
