from fastapi import APIRouter

# api
from api.v2.endpoints.recommendations import recommendations_router

def api_v2_router():
    router = APIRouter()

    router.include_router(recommendations_router())

    return router
