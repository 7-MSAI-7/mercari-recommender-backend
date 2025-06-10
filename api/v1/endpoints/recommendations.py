from typing import List
from fastapi import APIRouter

# schemas
from schemas.customer_sequence import CustomerSequence

# services
from services.data_loader import intialize_merrec_dataframe
from services.model_loader import (
    initialize_idx_to_item_id,
    initialize_trained_model,
)
from services.recommendation_service import generate_recommendations

# core
from core.config import EVENT_TO_IDX

# Load data and models
# 1-1. Azure에서 데이터프레임 로드 및 전처리
merrec_dataframe = intialize_merrec_dataframe()

# 1-2. 아이템 ID-인덱스 매핑 생성
idx_to_item_id = initialize_idx_to_item_id(merrec_dataframe)

# 1-3. 사전 학습된 GRU 모델 로드
trained_model = initialize_trained_model(
    n_events=len(EVENT_TO_IDX), n_items=len(idx_to_item_id)
)


def recommendations_router():
    """
    추천 API를 위한 APIRouter를 생성하고 반환합니다.

    이 함수는 애플리케이션 시작 시 로드된 데이터와 모델을 클로저(closure) 형태로
    내부의 API 엔드포인트 함수에서 사용할 수 있도록 캡슐화합니다.
    이를 통해 매 요청마다 데이터와 모델을 로드하는 오버헤드를 방지합니다.

    Returns:
        APIRouter: '/recommendations' 엔드포인트가 정의된 FastAPI 라우터 객체.
    """
    router = APIRouter()

    @router.post(
        "/recommendations",
        summary="사용자 시퀀스 기반 상품 추천",
        description="사용자의 상품 조회, 좋아요 등 행동 시퀀스를 입력받아 다음에 구매할 확률이 높은 상위 20개 상품을 추천합니다.",
    )
    def read_recommendations(
        sequences: List[CustomerSequence],
    ):
        """
        사용자 행동 시퀀스를 받아 추천 아이템 목록을 생성하는 API 엔드포인트입니다.

        Args:
            sequences (List[CustomerSequence]): 사용자의 행동 시퀀스 데이터.
                                                 Pydantic 모델 `CustomerSequence`의 리스트 형태여야 합니다.

        Returns:
            List[dict]: 추천된 아이템 목록과 각 아이템의 정보(점수 포함).
        """
        recommendations = generate_recommendations(
            dataframe=merrec_dataframe,
            trained_model=trained_model,
            idx_to_item_id=idx_to_item_id,
            customer_sequences=sequences,
        )

        return recommendations

    return router 