from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: EmailStr

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True

class BaseImageResponse(BaseModel):
    image_id: int
    collection_id: int
    image_url: str
    upload_order: int
    uploaded_at: datetime

    class Config:
        from_attributes = True

class CollectionCreate(BaseModel):
    title: str

class CollectionResponse(BaseModel):
    collection_id: int
    user_id: int
    title: str
    pattern_image_url: Optional[str]
    created_at: datetime
    base_images: List[BaseImageResponse] = []

    class Config:
        from_attributes = True

class ItemCreate(BaseModel):
    item_type: str
    item_3d_model_url: str
    thumbnail_url: Optional[str]

class ItemResponse(BaseModel):
    item_id: int
    item_type: str
    item_3d_model_url: str
    thumbnail_url: Optional[str]

    class Config:
        from_attributes = True