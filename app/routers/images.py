from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
import io
import os

import trimesh
import xatlas
import numpy as np

router = APIRouter()


@router.post(
    "/apply-uv",
    summary="Generate UV maps for a .glb model and return the UV-fixed .glb",
    response_description="UV-fixed GLB file",
)
async def apply_uv(
    model: UploadFile = File(..., description="GLB model file (.glb)"),
):
    """
    Upload a .glb file. Meshes that already have UV coordinates are kept as-is;
    meshes without UV coordinates get UV maps generated via xatlas.
    Returns the fully UV-mapped model as a downloadable .glb file.
    """
    if not model.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Uploaded model must be a .glb file.")

    model_bytes = await model.read()

    with tempfile.TemporaryDirectory() as td:
        # Write uploaded GLB to disk so trimesh can load it
        glb_in = os.path.join(td, "input.glb")
        with open(glb_in, "wb") as f:
            f.write(model_bytes)

        try:
            scene = trimesh.load(glb_in)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cannot load model: {exc}")

        new_geometry = {}
        for name, mesh in scene.geometry.items():
            has_uv = (
                hasattr(mesh.visual, "uv")
                and mesh.visual.uv is not None
                and len(mesh.visual.uv) > 0
            )

            if has_uv:
                new_geometry[name] = mesh
                continue

            # Generate UV coordinates with xatlas
            try:
                v = mesh.vertices.astype(np.float32)
                f = mesh.faces.astype(np.uint32)

                atlas = xatlas.Atlas()
                atlas.add_mesh(v, f)
                atlas.generate()

                vmapping, indices, uvs = atlas[0]

                new_mesh = trimesh.Trimesh(
                    vertices=v[vmapping],
                    faces=indices,
                    process=False,
                )
                new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
                new_geometry[name] = new_mesh
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"UV generation failed for mesh '{name}': {exc}",
                )

        fixed_scene = trimesh.Scene(new_geometry)

        glb_out = os.path.join(td, "uv_fixed.glb")
        try:
            fixed_scene.export(glb_out)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

        with open(glb_out, "rb") as f:
            glb_bytes = f.read()

    return StreamingResponse(
        io.BytesIO(glb_bytes),
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "attachment; filename=uv_fixed.glb"},
    )
