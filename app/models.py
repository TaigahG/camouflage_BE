from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from .database import Base

# User management is handled by Supabase auth.users.
# user_id is a UUID string referencing auth.users.id.

class Collection(Base):
    __tablename__ = "collections"

    collection_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # UUID from auth.users
    title = Column(String, index=True, nullable=False)
    pattern_image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    base_images = relationship("BaseImage", back_populates="collection", cascade="all, delete-orphan")
    applied_patterns = relationship("AppliedPattern", back_populates="collection", cascade="all, delete-orphan")

class BaseImage(Base):
    __tablename__ = "base_images"

    image_id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.collection_id", ondelete="CASCADE"), nullable=False)
    image_url = Column(String, nullable=False)
    upload_order = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    collection = relationship("Collection", back_populates="base_images")

class Item(Base):
    __tablename__ = "items"
    
    item_id = Column(Integer, primary_key=True, index=True)
    item_type = Column(String, nullable=False)
    item_3d_model_url = Column(String, nullable=False)
    thumbnail_url = Column(String, nullable=True)

class AppliedPattern(Base):
    __tablename__ = "applied_patterns"
    
    applied_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # UUID from auth.users
    collection_id = Column(Integer, ForeignKey("collections.collection_id", ondelete="CASCADE"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.item_id", ondelete="SET NULL"), nullable=True)  # Can be null if item deleted
    
    applied_model_url = Column(String, nullable=False)  
    thumbnail_url = Column(String, nullable=True) 
    
    title = Column(String, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)
    
    collection = relationship("Collection", back_populates="applied_patterns")