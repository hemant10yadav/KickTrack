from fastapi import FastAPI

from app.api.routes import health, videos
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(videos.router, prefix="/api", tags=["videos"])


@app.get("/")
def root():
    return {"message": f"{settings.app_name} API"}
