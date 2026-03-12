from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
import uuid


class UserInfo(BaseModel):
    """Authenticated user info extracted from the Supabase JWT."""
    id: uuid.UUID  # UUID from auth.users
    email: str


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
    user_id: uuid.UUID  # UUID from auth.users
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

class AppliedPatternCreate(BaseModel):
    collection_id: int
    item_id: int
    title: Optional[str] = None

class AppliedPatternResponse(BaseModel):
    applied_id: int
    user_id: int
    collection_id: int
    item_id: Optional[int]
    applied_model_url: str
    thumbnail_url: Optional[str]
    title: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True