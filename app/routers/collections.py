from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas
from ..database import get_db
from ..auth import get_current_user
from ..schemas import UserInfo
from ..storage import upload_base_image, upload_pattern, delete_base_images, delete_pattern


def _get_pattern_service():
    """Lazy import to avoid crashing if AI deps (torch/diffusers) aren't installed."""
    try:
        from ..services.pattern_service import pattern_service
        return pattern_service
    except ImportError:
        return None

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("/", response_model=schemas.CollectionResponse, status_code=201)
async def create_collection(
    title: str = Form(...),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    
    '''
    Docstring for create_collection
    
    :param user_id: Description
    :type user_id: int
    :param title: Description
    :type title: str
    :param images: Description
    :type images: List[UploadFile]
    :param db: Description
    :type db: Session
    '''

    """
    1. Validate user exists
    2. Validate 1-9 images
    3. Validate image types
    4. Create collection in database
    5. Upload each image to Supabase Storage
    6. Save image URLs to database
    7. Generate AI camouflage pattern (or fallback to dummy)
    8. Upload the generated pattern to Supabase Storage and save URL to database
    9. Return Collection Data (Generated Pattern URL, and Pattern Name)
    """
    user_id = current_user.id
    if not(1<= len(images) <= 9):
        raise HTTPException(status_code=404, detail="Must upload between 1-9 images")
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    for i in images:
        if i.content_type not in allowed_types:
            raise HTTPException(
                status_code=404,
                detail=f"Invalid file type: {i.content_type}. Allowed: jpeg, jpg, png, webp"
            )
        
    collection_data = schemas.CollectionCreate(title=title)
    db_collection = crud.create_collections(db, user_id, collection_data)

    for idx, image in enumerate(images, start=1):
        content = await image.read()

        try:
            image_url = upload_base_image(
                file_content=content,
                filename=image.filename,
                user_id=user_id,
                collection_id=db_collection.collection_id
            )
        except Exception:
            crud.delete_collections(db, db_collection.collection_id)
            raise HTTPException(status_code=500, detail="Failed to upload images to storage")
        
        crud.create_image(db, db_collection.collection_id, image_url, idx)

    # ─────────────────────────────────────────
    # AI PATTERN GENERATION
    # ─────────────────────────────────────────
    # If AI is available, generate the pattern now.
    # If not, the collection is saved without a pattern —
    # the user can retry via POST /collections/{id}/generate.
    pattern_service = _get_pattern_service()

    if pattern_service and pattern_service.is_ready:
        try:
            db_images = crud.get_collection_images(db, db_collection.collection_id)
            image_urls = [img.image_url for img in db_images]

            pattern_bytes = await pattern_service.generate_pattern(
                image_urls=image_urls,
                apply_segmentation=True
            )

            if pattern_bytes:
                pattern_url = upload_pattern(
                    file_content=pattern_bytes,
                    collection_id=db_collection.collection_id,
                    user_id=user_id
                )
                crud.update_collection_pattern(db, db_collection.collection_id, pattern_url)
        except Exception as e:
            print(f"[Collections] AI generation failed: {e}")

    db.refresh(db_collection)
    return db_collection



@router.post("/{collection_id}/generate", response_model=schemas.CollectionResponse)
async def regenerate_pattern(
    collection_id: int,
    apply_segmentation: bool = True,
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Regenerate the camouflage pattern for an existing collection.

    Use cases:
        - AI wasn't available when collection was created
        - User wants to try with/without SLIC segmentation
        - User wants a fresh generation (different random seed)

    Query params:
        apply_segmentation=false → Raw SD3 output (like Gen1 in the thesis)
        apply_segmentation=true  → SLIC segmented output (like Seg1 in the thesis)
    """
    pattern_service = _get_pattern_service()
    if not pattern_service or not pattern_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="AI model is not available. Please try again later."
        )

    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if db_collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get base image URLs
    db_images = crud.get_collection_images(db, db_collection.collection_id)
    image_urls = [img.image_url for img in db_images]

    if not image_urls:
        raise HTTPException(status_code=400, detail="Collection has no base images")

    # Delete old pattern from storage
    if db_collection.pattern_image_url:
        delete_pattern(user_id=current_user.id, collection_id=collection_id)

    # Generate new pattern
    pattern_bytes = await pattern_service.generate_pattern(
        image_urls=image_urls,
        apply_segmentation=apply_segmentation
    )

    if not pattern_bytes:
        raise HTTPException(status_code=500, detail="Pattern generation failed")

    # Upload and save
    pattern_url = upload_pattern(
        file_content=pattern_bytes,
        collection_id=collection_id,
        user_id=current_user.id
    )
    crud.update_collection_pattern(db, collection_id, pattern_url)

    db.refresh(db_collection)
    return db_collection


@router.get("/me", response_model=List[schemas.CollectionResponse])
def get_my_collections(
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Get all collections belonging to the authenticated user."""
    return crud.get_user_collections(db, user_id=current_user.id)

@router.get("/{collection_id}", response_model=schemas.CollectionResponse)
def get_collection(collection_id: int, db:Session = Depends(get_db)):
    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=404, detail="collection not found")
    return db_collection

@router.patch("/{collection_id}", response_model=schemas.CollectionResponse)
def rename_collection(
    collection_id: int,
    payload: schemas.CollectionUpdate,
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Rename a collection (title only)."""
    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if db_collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    new_title = payload.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    return crud.update_collection_title(db, collection_id, new_title)


@router.delete("/{collection_id}", status_code=204)
def delete_collection(
    collection_id: int,
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not db_collection:
        raise HTTPException(status_code=404, detail="collection not found")
    if db_collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this collection")

    delete_base_images(user_id=db_collection.user_id, collection_id=collection_id)

    if db_collection.pattern_image_url:
        delete_pattern(user_id=db_collection.user_id, collection_id=collection_id)

    crud.delete_collections(db, collection_id=collection_id)
    return None


# ──────────────────────────────────────────────────────────
# TEMPORARY TEST ENDPOINT — Remove before production
# ──────────────────────────────────────────────────────────
import uuid as _uuid

@router.post("/test-create", response_model=schemas.CollectionResponse, status_code=201)
async def test_create_collection(
    title: str = Form(...),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Temporary endpoint that bypasses auth for testing. DELETE before production."""
    user_id = _uuid.UUID("eb57a483-5be3-49f9-8f17-a55645ee8a4c")

    if not(1 <= len(images) <= 9):
        raise HTTPException(status_code=400, detail="Must upload between 1-9 images")
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    for i in images:
        if i.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {i.content_type}. Allowed: jpeg, jpg, png, webp"
            )

    collection_data = schemas.CollectionCreate(title=title)
    db_collection = crud.create_collections(db, user_id, collection_data)

    for idx, image in enumerate(images, start=1):
        content = await image.read()
        try:
            image_url = upload_base_image(
                file_content=content,
                filename=image.filename,
                user_id=user_id,
                collection_id=db_collection.collection_id
            )
        except Exception:
            crud.delete_collections(db, db_collection.collection_id)
            raise HTTPException(status_code=500, detail="Failed to upload images to storage")
        crud.create_image(db, db_collection.collection_id, image_url, idx)

    pattern_service = _get_pattern_service()
    if pattern_service and pattern_service.is_ready:
        try:
            db_images = crud.get_collection_images(db, db_collection.collection_id)
            image_urls = [img.image_url for img in db_images]
            pattern_bytes = await pattern_service.generate_pattern(
                image_urls=image_urls,
                apply_segmentation=True
            )
            if pattern_bytes:
                pattern_url = upload_pattern(
                    file_content=pattern_bytes,
                    collection_id=db_collection.collection_id,
                    user_id=user_id
                )
                crud.update_collection_pattern(db, db_collection.collection_id, pattern_url)
        except Exception as e:
            print(f"[Test] AI generation failed: {e}")

    db.refresh(db_collection)
    return db_collection