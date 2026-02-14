from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas
from ..database import get_db

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=schemas.UserResponse, status_code=201)
def create_user(user: schemas.UserCreate, db:Session = Depends(get_db)):
    db_user = crud.get_user_username(db, username=user)
    db_email = crud.get_user_email(db, email=user.email)
    if(db_user or db_email):
        return HTTPException(status_code=400, detail="Username/Email has already registered")
    
    return crud.create_user(db=db, user=user)

@router.get("/{user_id}", response_model=schemas.UserResponse)
def get_user(user_id:int, db:Session = Depends[get_db]):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None: 
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/", response_model=List[schemas.UserResponse])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users