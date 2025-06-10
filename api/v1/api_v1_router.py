from fastapi import APIRouter

# api
from api.v1.endpoints.recommendations import recommendations_router


def api_v1_router():
    router = APIRouter()

    router.include_router(recommendations_router())

    return router
