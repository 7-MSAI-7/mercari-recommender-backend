from fastapi import APIRouter

# api
from api.v1.endpoints.recommendations import recommendations_router
from api.v1.endpoints.customer_behaviors import customer_behaviors_router
from api.v1.endpoints.products import products_router

def api_v1_router():
    router = APIRouter()

    router.include_router(recommendations_router())
    router.include_router(customer_behaviors_router(), prefix="/customers", tags=["customers"])
    router.include_router(products_router(), prefix="/products", tags=["products"])

    return router
