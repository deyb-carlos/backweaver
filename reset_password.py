from fastapi import BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr

conf = ConnectionConfig(
    MAIL_USERNAME="sceneweaver840@gmail.com",
    MAIL_PASSWORD="iewa jtnk cusn eszg",
    MAIL_FROM="sceneweaver840@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
)

async def send_reset_email(email: str, token: str):
    # Create reset link (frontend URL)
    reset_link = f"http://35.213.136.241:5173/reset-password?token={token}"
    
    message = MessageSchema(
        subject="Password Reset Request",
        recipients=[email],
        body=f"""
        You requested a password reset. Click the link below to reset your password:
        
        {reset_link}
        
        This link will expire in 1 hour.
        
        If you didn't request this, please ignore this email.
        """,
        subtype="plain"
    )
    
    fm = FastMail(conf)
    await fm.send_message(message)