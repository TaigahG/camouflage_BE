from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    collections = relationship("Collection", back_populates="user", cascade="all, delete-orphan")

class Collection(Base):
    __tablename__ = "collections"

    collection_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    title = Column(String, index=True, nullable=False)
    pattern_image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="collections")
    base_images = relationship("BaseImage", back_populates="collection", cascade="all, delete-orphan")

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
