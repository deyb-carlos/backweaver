from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from database import  Base
from datetime import datetime, timezone


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    storyboards = relationship("Storyboard", back_populates="owner")


class Storyboard(Base):
    __tablename__ = "storyboards"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    thumbnail = Column(String, index=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    owner = relationship("User", back_populates="storyboards")
    images = relationship(
        "Image", back_populates="storyboard", cascade="all, delete-orphan"
    )


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    storyboard_id = Column(Integer, ForeignKey("storyboards.id"))
    image_path = Column(String)
    caption = Column(String)

    storyboard = relationship("Storyboard", back_populates="images")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


