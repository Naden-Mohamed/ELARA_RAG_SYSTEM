from fastapi import APIRouter
base = APIRouter(tags=["api"], prefix="/api")

@base.get('/')
def get_status():
    return "Healthy"

@base.get('/health')
def health_check():
    return {"status": "FastAPI is running"} 