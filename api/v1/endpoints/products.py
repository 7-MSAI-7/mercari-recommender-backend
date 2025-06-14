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
        results = await asyncio.gather(task)

        return results[0]
    
    return router
