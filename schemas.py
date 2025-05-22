from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional


class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool
    storyboard_ids: List[int] = []  # Just include a list of storyboard IDs
    model_config = {"from_attributes": True}


class StoryboardOut(BaseModel):
    id: int
    name: str
    owner_id: int
    # owner: Optional[UserOut] = None  # Remove or comment out the nested UserOut
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    thumbnail: str
    image_ids: List[int] = []  # Just include a list of image IDs
    model_config = {"from_attributes": True}


class ImageOut(BaseModel):
    id: int
    image_path: str
    caption: str
    storyboard_id: int
    # storyboard: Optional[StoryboardOut] = None # Remove or comment out the nested StoryboardOut
    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class StoryboardCreate(BaseModel):
    name: str
    owner_id: int


class StoryboardCreateNoOwner(BaseModel):
    name: str  # Used in the new endpoints


class ImageCreate(BaseModel):
    storyboard_id: int


class ResetTokenCreate(BaseModel):
    email: EmailStr
    token: str
    expires_at: datetime
