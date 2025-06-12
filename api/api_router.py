from fastapi import APIRouter

# api
from api.v1.api_v1_router import api_v1_router
from api.v2.api_v2_router import api_v2_router

def api_router():
    router = APIRouter()

    router.include_router(api_v1_router(), prefix="/v1")
    router.include_router(api_v2_router(), prefix="/v2")

    return router   