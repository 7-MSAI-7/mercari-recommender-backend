from fastapi import APIRouter, Request
from schemas.customer_behavior import CustomerBehavior
from services.customer_behavior_service import CustomerBehaviorService

def customer_behaviors_router():
    router = APIRouter()

    @router.post("/behaviors", response_model=list[CustomerBehavior])
    def store_customer_behavior(
        request: Request,
        behavior: CustomerBehavior
    ):
        """
        사용자 행동(상품 조회, 좋아요)을 세션에 저장합니다.
        """
        return CustomerBehaviorService.store_customer_behavior(
            request=request,
            name=behavior.name,
            event=behavior.event
        )

    @router.get("/behaviors", response_model=list[CustomerBehavior])
    def get_customer_behavior(
        request: Request
    ):
        """
        세션에 저장된 사용자 행동을 조회합니다.
        """
        return CustomerBehaviorService.get_behaviors(request=request)

    return router