from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas
from ..database import get_db
from ..auth import get_current_user
from ..schemas import UserInfo

router = APIRouter(prefix="/applied-patterns", tags=["applied-patterns"])


@router.get("/", response_model=List[schemas.AppliedPatternResponse])
def list_user_applied_patterns(
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Get all applied patterns for the current user"""
    return crud.get_user_applied_patterns(db, current_user.id)


@router.get("/{applied_id}", response_model=schemas.AppliedPatternResponse)
def get_applied_pattern(
    applied_id: int,
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Get a specific applied pattern (must own it)"""
    db_applied = crud.get_applied_pattern(db, applied_id)
    if not db_applied:
        raise HTTPException(status_code=404, detail="Applied pattern not found")
    if db_applied.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return db_applied


@router.get("/collection/{collection_id}", response_model=List[schemas.AppliedPatternResponse])
def list_collection_applied_patterns(
    collection_id: int,
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Get all applied patterns for a collection"""
    # Verify collection exists and belongs to user
    db_collection = crud.get_collections(db, collection_id)
    if not db_collection or db_collection.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Collection not found or access denied")
    
    return crud.get_collection_applied_patterns(db, collection_id)


@router.delete("/{applied_id}", status_code=204)
def delete_applied_pattern(
    applied_id: int,
    db: Session = Depends(get_db),
    current_user: UserInfo = Depends(get_current_user),
):
    """Delete an applied pattern (must own it)"""
    db_applied = crud.get_applied_pattern(db, applied_id)
    if not db_applied:
        raise HTTPException(status_code=404, detail="Applied pattern not found")
    if db_applied.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = crud.delete_applied_pattern(db, applied_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete applied pattern")
    return None
