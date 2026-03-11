import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas
from ..database import get_db
from ..auth import get_current_user
from ..schemas import UserInfo
from ..storage import upload_base_image, upload_pattern, delete_base_images, delete_pattern

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
    7. (TODO) Trigger AI pattern generation
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

    # TODO: Replace with actual AI pattern generation logic
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    dummy_pattern_path = os.path.join(assets_dir, "pattern_1.jpg")
    with open(dummy_pattern_path, "rb") as f:
        pattern_bytes = f.read()

    pattern_url = upload_pattern(
        file_content=pattern_bytes,
        collection_id=db_collection.collection_id,
        user_id=user_id
    )
    crud.update_collection_pattern(db, db_collection.collection_id, pattern_url)

    db.refresh(db_collection)
    return db_collection

@router.get("/{collection_id}", response_model=schemas.CollectionResponse)
def get_collection(collection_id: int, db:Session = Depends(get_db)):
    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not collection_id:
        raise HTTPException(status_code=404, detail="collection not found")
    return db_collection

@router.get("/me", response_model=List[schemas.CollectionResponse])
def get_my_collections(
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Get all collections belonging to the authenticated user."""
    return crud.get_user_collections(db, user_id=current_user.id)

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
