from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas
from ..database import get_db

router = APIRouter(prefix="/items", tags=["items"])


@router.post("/", response_model=schemas.ItemResponse, status_code=201)
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    """Create a new AR item (admin only in production)"""
    return crud.create_item(db=db, item=item)


@router.get("/", response_model=List[schemas.ItemResponse])
def list_items(db: Session = Depends(get_db)):
    """Get all available AR items"""
    items = crud.get_items(db)
    return items


@router.get("/{item_id}", response_model=schemas.ItemResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    """Get a specific AR item"""
    db_item = crud.get_item(db, item_id)
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item


@router.delete("/{item_id}", status_code=204)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    """Delete an AR item (admin only in production)"""
    success = crud.delete_item(db, item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return None