from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas
from ..database import get_db
from ..storage import upload_image, delete_image

router = APIRouter(prefix="/collections", tags=["collections"])



@router.post("/", response_model=schemas.CollectionResponse, status_code=201)
async def create_collection(
    user_id: int = Form(...),
    title: str = Form(...),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
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
    """
    db_user = crud.get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
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
        content = image.read()

        image_url = await upload_image(
            file = content,
            filename=image.filename,
            folder="base_images"
        )

        if not image_url:
            crud.delete_collections(db, db_collection.collection_id)
            raise HTTPException(status_code=500, detail="Failed to upload images to storage")
        
        crud.create_image(db, db_collection.collection_id, image_url, idx)

    print(f"[STUB] Would generate pattern for collection {db_collection.collection_id}")
    print(f"[STUB] Base images uploaded: {len(images)}")
    print(f"[STUB] User ID: {user_id}, Title: {title}")
    
    db.refresh(db_collection)
    return db_collection

@router.get("/{collection_id}", response_model=schemas.CollectionResponse)
def get_collection(collection_id: int, db:Session = Depends(get_db)):
    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not collection_id:
        raise HTTPException(status_code=404, detail="collection not found")
    return db_collection

@router.get("/user/{user_id}", response_model=List[schemas.CollectionResponse])
def get_user_collection(user_id: int, db:Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="user not found")
    collection = crud.get_user_collections(db, user_id=user_id)
    return collection

@router.delete("/{collection_id}", status_code=204)
async def delete_collection(collection_id: int, db:Session = Depends(get_db)):
    db_collection = crud.get_collections(db, collection_id=collection_id)
    if not collection_id:
        raise HTTPException(status_code=404, detail="collection not found")
    
    for base_image in db_collection.base_images:
        await delete_image(base_image.image_url)

    if db_collection.pattern_image_url:
        await delete_image(db_collection.pattern_image_url)

    crud.delete_collections(db, collection_id=collection_id)
    return None
