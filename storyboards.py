from sqlalchemy.orm import Session
from datetime import datetime
from database import Base
from models import Storyboard
import random
import string
from fastapi import HTTPException
from datetime import timezone
from schemas import StoryboardCreate, StoryboardOut




def create_storyboard(db: Session, storyboard: StoryboardCreate):
    thumbnail_url = "https://sceneweaver.s3.ap-southeast-2.amazonaws.com/assets/tumblr_f62bc01ba9fb6acf8b5d438d6d2ae71a_c5a496d1_1280.jpg"
    db_storyboard = Storyboard(
        name=storyboard.name,
        thumbnail=thumbnail_url,
        owner_id=storyboard.owner_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(db_storyboard)
    db.commit()
    db.refresh(db_storyboard)
    return db_storyboard

def get_storyboard_by_name(db: Session, storyboard: StoryboardOut):
    return db.query(Storyboard).filter(Storyboard.name == storyboard.name).first()

