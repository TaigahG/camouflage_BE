"""
POST /api/retexture-clothes

Apply a collection's camouflage style to the upper-body clothing of a
user-supplied photo, using SegFormer clothes segmentation + SD3 inpainting.
Returns the retextured PNG image directly.
"""
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import crud
from ..auth import get_current_user
from ..database import get_db
from ..schemas import AppliedPatternResponse, UserInfo
from ..storage import upload_applied_outfit


router = APIRouter(tags=["retexture"])


def _get_retexture_service():
    """Lazy import — avoids crashing the app if AI deps aren't installed."""
    try:
        from ..services.retexture_service import retexture_service
        return retexture_service
    except ImportError:
        return None


@router.post(
    "/retexture-clothes",
    summary="Retexture the upper-body clothing in a photo using a collection's camo style",
    response_description="PNG image of the retextured photo",
)
async def retexture_clothes(
    photo: UploadFile = File(..., description="Person photo (JPEG / PNG / WebP)"),
    collection_id: int = Form(..., description="Collection ID to source the camo style from"),
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if photo.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{photo.content_type}'. Use JPEG, PNG, or WebP.",
        )

    db_collection = crud.get_collections(db, collection_id)
    if not db_collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if db_collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized for this collection")

    service = _get_retexture_service()
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="AI pipeline is not available. Try again later.",
        )

    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="Empty photo upload")

    db_images = crud.get_collection_images(db, collection_id)
    base_image_urls = [img.image_url for img in db_images]

    result_bytes = await service.retexture_upper_body(
        photo_bytes=photo_bytes,
        base_image_urls=base_image_urls,
    )

    if result_bytes is None:
        raise HTTPException(
            status_code=422,
            detail="Could not detect upper-body clothing in the photo, or generation failed.",
        )

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=retextured.png"},
    )


@router.post(
    "/retexture-outfit",
    summary="Apply a collection's camo style to a clothing-only image and persist the result",
    response_model=AppliedPatternResponse,
    status_code=201,
)
async def retexture_outfit(
    outfit: UploadFile = File(..., description="Clothing image (transparent or white background)"),
    collection_id: int = Form(..., description="Collection ID to source the camo style from"),
    outfit_type: str = Form("shirt", description="Outfit type (e.g. hoodie, t-shirt, jacket)"),
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if outfit.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{outfit.content_type}'. Use JPEG, PNG, or WebP.",
        )

    db_collection = crud.get_collections(db, collection_id)
    if not db_collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if db_collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized for this collection")

    service = _get_retexture_service()
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="AI pipeline is not available. Try again later.",
        )

    outfit_bytes_in = await outfit.read()
    if not outfit_bytes_in:
        raise HTTPException(status_code=400, detail="Empty outfit upload")

    db_images = crud.get_collection_images(db, collection_id)
    base_image_urls = [img.image_url for img in db_images]

    normalized_outfit_type = outfit_type.strip().lower() or "shirt"

    result_bytes = await service.retexture_outfit(
        outfit_bytes=outfit_bytes_in,
        base_image_urls=base_image_urls,
        outfit_type=normalized_outfit_type,
    )

    if result_bytes is None:
        raise HTTPException(
            status_code=422,
            detail="Could not detect clothing region, or generation failed.",
        )

    # Create the AppliedPattern row first so we have an applied_id for the path.
    db_applied = crud.create_applied_pattern(
        db=db,
        user_id=current_user.id,
        collection_id=collection_id,
        applied_model_url="placeholder",
        title=f"{normalized_outfit_type.capitalize()} — {db_collection.title}",
    )

    try:
        applied_url = upload_applied_outfit(
            file_content=result_bytes,
            user_id=current_user.id,
            collection_id=collection_id,
            applied_id=db_applied.applied_id,
            file_extension="png",
        )
        db_applied.applied_model_url = applied_url
        db.commit()
        db.refresh(db_applied)
    except Exception as e:
        db.delete(db_applied)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save patterned outfit: {e}",
        )

    return db_applied
