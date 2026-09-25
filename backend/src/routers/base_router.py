from fastapi import APIRouter

base = APIRouter(tags=["api"])


@base.get("/")
def get_status():
    return "Healthy"


@base.get("/healthz")
def health_check():
    return {"status": "FastAPI is running"}
