from fastapi import APIRouter, Depends

from ..auth import get_current_user
from ..schemas import UserInfo

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserInfo)
async def get_me(current_user: UserInfo = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user