import random
import asyncio
from fastapi import APIRouter, Request

# services
from services.data_loader import intialize_merrec_dataframe
from services.model_loader import (
    initialize_idx_to_item_id,
    initialize_trained_gru_model,
)
from services.recommendation_service import generate_recommendations_using_gru
from services.google_shopping_service import search_google_shopping
from services.customer_behavior_service import CustomerBehaviorService

# core
from core.config import EVENT_TO_IDX

# Load data and models
# 1-1. Azure에서 데이터프레임 로드 및 전처리
merrec_dataframe = intialize_merrec_dataframe()

# 1-2. 아이템 ID-인덱스 매핑 생성
idx_to_item_id = initialize_idx_to_item_id(merrec_dataframe)

# 1-3. 사전 학습된 모델 로드
trained_gru_model = initialize_trained_gru_model(
    n_events=len(EVENT_TO_IDX), n_items=len(idx_to_item_id)
)


def recommendations_router():
    """
    GRU 모델을 사용한 추천 API를 위한 APIRouter를 생성하고 반환합니다.

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
        recommendations = generate_recommendations_using_gru(
            dataframe=merrec_dataframe,
            trained_model=trained_gru_model,
            idx_to_item_id=idx_to_item_id,
            customer_behaviors=customer_behaviors,
        )

        if not customer_behaviors:
            return []

        # Google Shopping 검색을 위한 키워드 리스트 생성
        search_keywords = [customer_behaviors[-1]["name"]] + [
            rec["name"] for rec in random.choices(recommendations, k=3)
        ]

        # 모든 검색 작업을 비동기적으로 생성
        tasks = [search_google_shopping(keyword) for keyword in search_keywords]

        # 생성된 작업을 동시에 실행하고 결과 수집
        search_results = await asyncio.gather(*tasks)

        recommendation_products = []

        # 첫 번째 검색 결과 (마지막 행동 기반)에서는 최대 5개만 추가
        first_result = search_results[0]
        if first_result:
            recommendation_products.extend(first_result[:5])

        # 나머지 추천 검색 결과 추가
        for products in search_results[1:]:
            if products:
                recommendation_products.extend(products)

        return recommendation_products

    return router 