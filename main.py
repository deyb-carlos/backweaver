from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    BackgroundTasks,
    Form,
    UploadFile,
    File,
    APIRouter,
)
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from database import SessionLocal, engine
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from schemas import (
    UserCreate,
    StoryboardOut,
    StoryboardCreateNoOwner,
    ImageOut,
)
import auth, database, storyboards
from PIL import Image
from io import BytesIO
import random
from typing import List, Optional
import models
import secrets
import string
from reset_password import send_reset_email
from fastapi import BackgroundTasks
from batch_generator import generate_batch_images, generate_single_image
from s3 import delete_image_from_s3
from text_processor import get_resolved_sentences, get_script_captions

app = FastAPI()


origins = [
    "http://localhost:5173",
    "https://sceneweaver.site",
    "https://mung-bean-sceneweaver.hf.space",
    "sceneweaver.netlify.app",
    "ec2-3-106-55-36.ap-southeast-2.compute.amazonaws.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


@api_router.get("/")
async def root():
    return {"message": "Welcome to SceneWeaver"}


@api_router.post("/regenerate-image/{image_id}")
async def regenerate_image(
    image_id: int,
    caption: str = Form(...),
    seed: Optional[int] = Form(None),
    resolution: str = Form(...),
    isOpenPose: bool = Form(False),
    pose_img: UploadFile = File(None),
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        # Get the image with its associated storyboard
        db_image = (
            db.query(models.Image)
            .join(models.Storyboard)
            .filter(
                models.Image.id == image_id,
                models.Storyboard.owner_id == user.id,
            )
            .first()
        )

        if not db_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found or not owned by user",
            )
        pose_image_obj = None
        if isOpenPose and pose_img:
            image_data = await pose_img.read()
            pose_image_obj = Image.open(BytesIO(image_data))

        # Regenerate the image (this will maintain the same filename)
        generate_single_image(
            image_id, caption, seed, resolution, isOpenPose, pose_image_obj
        )

        # Update storyboard's updated_at timestamp
        storyboard = (
            db.query(models.Storyboard)
            .filter(models.Storyboard.id == db_image.storyboard_id)
            .first()
        )
        if storyboard:
            storyboard.updated_at = datetime.now(timezone.utc)
            db.commit()

        return {"message": "Image regenerated successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error regenerating image: {str(e)}",
        )


@api_router.patch("/images/{image_id}/caption")
async def update_image_caption(
    image_id: int,
    caption: str = Form(...),
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        # Get the image with its associated storyboard
        db_image = (
            db.query(models.Image)
            .join(models.Storyboard)
            .filter(
                models.Image.id == image_id,
                models.Storyboard.owner_id == user.id,
            )
            .first()
        )

        if not db_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found or not owned by user",
            )

        # Update the caption
        db_image.caption = caption

        # Update storyboard's updated_at timestamp
        storyboard = (
            db.query(models.Storyboard)
            .filter(models.Storyboard.id == db_image.storyboard_id)
            .first()
        )
        if storyboard:
            storyboard.updated_at = datetime.now(timezone.utc)

        db.commit()

        return {"message": "Caption updated successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating caption: {str(e)}",
        )


@api_router.get("/storyboard/images/{storyboard_id}", response_model=List[ImageOut])
async def get_storyboard_images(
    storyboard_id: int,
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Get storyboard with images using the relationship
        storyboards = (
            db.query(models.Storyboard)
            .filter(
                models.Storyboard.id == storyboard_id,
                models.Storyboard.owner_id == user.id,
            )
            .first()
        )

        if not storyboards:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Storyboard not found or access denied",
            )

        # Return just the images as a list of ImageOut objects
        return [
            ImageOut(
                id=image.id,
                image_path=image.image_path,
                caption=image.caption,
                storyboard_id=image.storyboard_id,
            )
            for image in storyboards.images
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching images: {str(e)}",
        )


@api_router.post("/register")
def register(user: UserCreate, db: Session = Depends(database.get_db)):
    db_user = auth.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    db_user_email = auth.get_user_by_email(db, user.email)
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already exists")

    try:
        created_user = auth.create_user(db=db, user=user)
        return {
            "id": created_user.id,
            "username": created_user.username,
            "email": created_user.email,
        }
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400, detail="Error while creating user. Please try again later."
        )


@api_router.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(database.get_db),
):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(days=7)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@api_router.post("/refresh-token")
def refresh_token(token: str = Depends(auth.oauth2_scheme)):
    try:
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(SessionLocal(), username)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        access_token_expires = timedelta(days=7)
        access_token = auth.create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@api_router.get("/verify-token")
async def verify_user_token(token: str = Depends(auth.oauth2_scheme)):
    auth.verify_token(token=token)
    return {"message": "Token is valid"}


@api_router.get("/me")
async def get_current_user(
    token: str = Depends(auth.oauth2_scheme), db: Session = Depends(database.get_db)
):
    try:
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@api_router.post("/home", response_model=StoryboardOut)
def create_storyboard(
    storyboard: StoryboardCreateNoOwner,
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:

        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        existing_storyboard = (
            db.query(models.Storyboard)
            .filter(
                models.Storyboard.owner_id == user.id,
                models.Storyboard.name == storyboard.name,
            )
            .first()
        )

        if existing_storyboard:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Storyboard with this name already exists",
            )

        db_storyboard = models.Storyboard(
            name=storyboard.name,
            owner_id=user.id,
            thumbnail="https://sceneweaver.s3.ap-southeast-2.amazonaws.com/assets/thumbnail.png",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db.add(db_storyboard)
        db.commit()
        db.refresh(db_storyboard)

        if db_storyboard.images:
            db_storyboard.thumbnail = db_storyboard.images[0].image_path
            db.commit()

        return db_storyboard

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating storyboard: {str(e)}",
        )


@api_router.patch("/storyboard/{storyboard_id}")
def rename_storyboard(
    storyboard_id: int,
    storyboard: StoryboardCreateNoOwner,
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        # Get existing storyboard
        db_storyboard = (
            db.query(models.Storyboard)
            .filter(
                models.Storyboard.id == storyboard_id,
                models.Storyboard.owner_id == user.id,
            )
            .first()
        )

        if not db_storyboard:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Storyboard not found or not owned by user",
            )

        # Update storyboard
        db_storyboard.name = storyboard.name
        db_storyboard.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(db_storyboard)
        return db_storyboard

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating storyboard: {str(e)}",
        )


@api_router.delete("/storyboard/{storyboard_id}")
def delete_storyboard(
    storyboard_id: int,
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        # Find and delete storyboard
        db_storyboard = (
            db.query(models.Storyboard)
            .filter(
                models.Storyboard.id == storyboard_id,
                models.Storyboard.owner_id == user.id,
            )
            .first()
        )

        if not db_storyboard:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Storyboard not found or not owned by user",
            )

        for image in db_storyboard.images:
            delete_image_from_s3(image.image_path)

        db.delete(db_storyboard)
        db.commit()
        return {"message": "Storyboard deleted successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting storyboard: {str(e)}",
        )


@api_router.get("/home", response_model=List[StoryboardOut])
def get_user_storyboards(
    db: Session = Depends(database.get_db), token: str = Depends(auth.oauth2_scheme)
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Fetch storyboards with images eager-loaded
        storyboards = (
            db.query(models.Storyboard)
            .options(joinedload(models.Storyboard.images))
            .filter(models.Storyboard.owner_id == user.id)
            .all()
        )

        # Set thumbnail to newest image (highest id)
        for storyboard in storyboards:
            if storyboard.images:
                newest_image = max(storyboard.images, key=lambda img: img.id)
                if storyboard.thumbnail != newest_image.image_path:
                    storyboard.thumbnail = newest_image.image_path
                    storyboard.updated_at = datetime.now(timezone.utc)

        db.commit()

        return storyboards or []

    except Exception as e:
        print(f"Unexpected error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching storyboards: {str(e)}",
        )


@api_router.get("/storyboard/{storyboard_id}/{name}", response_model=StoryboardOut)
def get_storyboard(
    storyboard_id: int,
    name: str,
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        # Get the storyboard
        db_storyboard = (
            db.query(models.Storyboard)
            .filter(
                models.Storyboard.id == storyboard_id,
                models.Storyboard.owner_id == user.id,
                models.Storyboard.name == name,
            )
            .first()
        )

        if not db_storyboard:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Storyboard not found or not owned by user",
            )

        return db_storyboard

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching storyboard: {str(e)}",
        )


@api_router.delete("/images/{image_id}")
def delete_image(
    image_id: int,
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    try:
        # Verify token and get current user
        username = auth.verify_token_string(token)
        user = auth.get_user_by_username(db, username)

        # Get the image with its associated storyboard
        db_image = (
            db.query(models.Image)
            .join(models.Storyboard)
            .filter(
                models.Image.id == image_id,
                models.Storyboard.owner_id == user.id,
            )
            .first()
        )

        if not db_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found or not owned by user",
            )

        # Delete the image from S3
        delete_image_from_s3(db_image.image_path)

        # Delete the image from database
        db.delete(db_image)
        db.commit()

        # Update storyboard thumbnail if needed
        storyboard = (
            db.query(models.Storyboard)
            .filter(models.Storyboard.id == db_image.storyboard_id)
            .first()
        )

        if storyboard:
            # If the deleted image was the thumbnail, update it to the newest remaining image
            if storyboard.thumbnail == db_image.image_path:
                remaining_images = storyboard.images
                if remaining_images:
                    newest_image = max(remaining_images, key=lambda img: img.id)
                    storyboard.thumbnail = newest_image.image_path
                else:
                    storyboard.thumbnail = "https://sceneweaver.s3.ap-southeast-2.amazonaws.com/assets/thumbnail.png"

                storyboard.updated_at = datetime.now(timezone.utc)
                db.commit()

        return {"message": "Image deleted successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting image: {str(e)}",
        )


@api_router.post("/forgot-password")
async def forgot_password(
    background_tasks: BackgroundTasks,
    username: str = Form(...),
    db: Session = Depends(database.get_db),
):

    user = auth.get_user_by_username(db, username)
    if not user:
        return {"message": "A reset link has been sent"}

    alphabet = string.ascii_letters + string.digits
    token = "".join(secrets.choice(alphabet) for _ in range(32))

    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    db_token = models.PasswordResetToken(
        email=user.email, token=token, expires_at=expires_at
    )
    db.add(db_token)
    db.commit()

    background_tasks.add_task(send_reset_email, email=user.email, token=token)

    return {"message": "A reset link has been sent"}


@api_router.post("/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(database.get_db),
):
    # Verify token
    db_token = (
        db.query(models.PasswordResetToken)
        .filter(
            models.PasswordResetToken.token == token,
            models.PasswordResetToken.expires_at > datetime.now(timezone.utc),
        )
        .first()
    )

    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token"
        )

    # Get user by email
    user = auth.get_user_by_email(db, db_token.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found"
        )

    # Update password
    hashed_password = auth.get_password_hash(new_password)
    user.hashed_password = hashed_password
    db.commit()

    # Delete the used token
    db.delete(db_token)
    db.commit()

    return {"message": "Password updated successfully"}


@api_router.post("/generate-images/{storyboard_id}")
async def generate_images(
    background_tasks: BackgroundTasks,
    storyboard_id: int,
    story: str = Form(...),
    resolution: str = Form("1:1"),
    isStory: bool = Form(True),
    db: Session = Depends(database.get_db),
    token: str = Depends(auth.oauth2_scheme),
):
    username = auth.verify_token_string(token)
    user = auth.get_user_by_username(db, username)

    storyboard = (
        db.query(models.Storyboard)
        .filter_by(id=storyboard_id, owner_id=user.id)
        .first()
    )
    if not storyboard:
        raise HTTPException(status_code=404, detail="Storyboard not found")

    storyboard.updated_at = datetime.now(timezone.utc)
    db.commit()

    if isStory:
        caption_length = get_resolved_sentences(story)
    else:
        caption_length = get_script_captions(story)

    background_tasks.add_task(
        generate_batch_images, story, storyboard.id, resolution, isStory
    )

    if storyboard.images:
        sorted_images = sorted(storyboard.images, key=lambda img: img.id, reverse=True)
        newest_image = sorted_images[0]
        storyboard.thumbnail = newest_image.image_path
        db.commit()

    return {"message": "Image generation started", "count": len(caption_length)}


app.include_router(api_router)
