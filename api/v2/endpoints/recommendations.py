import asyncio
from fastapi import APIRouter, Request


# services
from services.model_loader import (
    initialize_trained_two_tower_model,
)
from services.google_shopping_service import search_google_shopping
from services.customer_behavior_service import CustomerBehaviorService
from services.recommendation_service import generate_recommendations_using_two_tower


trained_two_tower_model, two_tower_mappings, two_tower_df = initialize_trained_two_tower_model()


def recommendations_router():
    """
    Two-Tower 모델을 사용한 추천 API를 위한 APIRouter를 생성하고 반환합니다.

    이 함수는 애플리케이션 시작 시 로드된 데이터와 모델을 클로저(closure) 형태로
    내부의 API 엔드포인트 함수에서 사용할 수 있도록 캡슐화합니다.
    이를 통해 매 요청마다 데이터와 모델을 로드하는 오버헤드를 방지합니다.

    Returns:
        APIRouter: '/recommendations' 엔드포인트가 정의된 FastAPI 라우터 객체.
    """
    router = APIRouter()

    @router.post(
        "/recommendations",
        summary="사용자 세션 기반 상품 추천",
        description="사용자의 상품 조회, 좋아요 등 세션에 저장된 사용자 행동에 따라 상품을 추천합니다.",
    )
    async def create_recommendations(
        request: Request,
    ):
        """
        세션에 저장된 사용자 행동을 통해 추천 상품 목록을 생성하는 API 엔드포인트입니다.

        Returns:
            List[dict]: 추천된 아이템 목록과 각 아이템의 정보(점수 포함).
        """
        customer_behaviors = CustomerBehaviorService.get_behaviors(request)[-40:]
        if len(customer_behaviors) == 0:
            item_indices = two_tower_df.sample(3)["item_idx"].tolist()
            item_names = [two_tower_mappings["item_titles"][item_idx] for item_idx in item_indices]
            customer_behaviors = [ { "name": item_name } for item_name in item_names ][:4]
        print(customer_behaviors)

        keywords = generate_recommendations_using_two_tower(customer_behaviors, trained_two_tower_model, two_tower_mappings)
        print(keywords)

        # 모든 검색 작업을 비동기적으로 생성
        tasks = [search_google_shopping(keyword) for keyword in keywords[:4]]

        # 생성된 작업을 동시에 실행하고 결과 수집
        recommendation_products = await asyncio.gather(*tasks)

        return recommendation_products

    return router 