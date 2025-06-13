from fastapi import APIRouter

from services.google_shopping_service import search_google_shopping
import asyncio


def products_router():
    router = APIRouter()

    @router.get("/products")
    async def get_products(
        q: str,
    ):
        task = search_google_shopping(q)
        result = await asyncio.gather(task)

        return result
    
    return router
