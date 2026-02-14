from sqlalchemy.orm import Session
from . import models, schemas
from typing import List, Optional

#User
def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
def get_user(db:Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.user_id == user_id).first()
def get_user_username(db:Session, username: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.username == username).first()
def get_user_email(db:Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()
def get_users(db:Session, skip:0, limit:100):
    return db.query(models.User).offset(skip).limit(limit).all()


#Collections
def create_collections(db:Session, user_id:int, collection:schemas.CollectionCreate) -> models.Collection:
    db_collection = models.Collection(
        user_id = user_id,
        title = collection.title,
        pattern_img_url = None
    )
    db.add(db_collection)
    db.commit()
    db.refresh()
    return db_collection
def get_collections(db:Session, collection_id: int) -> Optional[models.Collection]:
    return db.query(models.Collection).filter(models.Collection.collection_id == collection_id)
def get_user_collections(db:Session, user_id: int) -> Optional[models.Collection]:
    return db.query(models.Collection).filter(
        models.Collection.user_id == user_id
    ).order_by(models.Collection.created_at.desc()).all()
def update_collection_pattern(db:Session, collection_id: int, pattern_url:str) -> Optional[models.Collection]:
    db_collection = get_collections(db, collection_id)
    if db_collection:
        db_collection.pattern_image_url = pattern_url
        db.commit()
        db.refresh(db_collection)
    return db_collection
def delete_collections(db:Session, collection_id:int):
    db_collection = get_collections(db, collection_id)
    if db_collection:
        db.delete(db_collection)
        db.commit()
        return True
    return False

#Base Image
def create_image(db:Session, collection_id:int, img:str, order:int):
    db_img = models.BaseImage(
        collection_id = collection_id,
        img = img,
        upload_order = order
    )
    db.add(db_img)
    db.commit
    db.refresh()
    return db_img
def get_collection_images(db:Session, collection_id:int):
    return db.query(models.BaseImage).filter(models.BaseImage.collection_id == collection_id).order_by(models.BaseImage.upload_order).all()

#items
def create_item(db: Session, item: schemas.ItemCreate):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
def get_items(db: Session):
    return db.query(models.Item).all()
def get_item(db: Session, item_id: int):
    return db.query(models.Item).filter(models.Item.item_id == item_id).first()
def delete_item(db: Session, item_id: int):
    db_item = get_item(db, item_id)
    if db_item:
        db.delete(db_item)
        db.commit()
        return True
    return False