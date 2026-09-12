import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


@router.post("/videos/upload")
def upload_video(file: UploadFile):
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(uuid.uuid4())
    destination = upload_dir / f"{video_id}{extension}"

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"video_id": video_id, "filename": file.filename}
