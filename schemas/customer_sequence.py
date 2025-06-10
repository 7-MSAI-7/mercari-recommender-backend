from enum import Enum
from pydantic import BaseModel, Field


class Event(str, Enum):
    """
    사용자가 수행할 수 있는 행동(이벤트)의 종류를 정의하는 열거형(Enum) 클래스입니다.
    API 요청 시 event 필드는 여기에 정의된 값 중 하나여야 합니다.
    """

    ITEM_VIEW = "item_view"
    ITEM_LIKE = "item_like"
    ITEM_ADD_TO_CART_TAP = "item_add_to_cart_tap"
    OFFER_MAKE = "offer_make"
    BUY_START = "buy_start"
    BUY_COMPLETE = "buy_comp"


class CustomerSequence(BaseModel):
    """
    API를 통해 입력받는 개별 사용자 행동 시퀀스 데이터의 구조를 정의하는 Pydantic 모델입니다.
    FastAPI는 이 모델을 사용하여 요청 본문의 유효성을 검사하고 데이터를 파싱합니다.
    """

    # 상품명: 사용자가 상호작용한 상품의 이름입니다.
    name: str = Field(..., description="상품명", example="T-shirt")

    # 이벤트: 사용자가 수행한 행동의 종류입니다.
    event: Event = Field(..., description="사용자 이벤트", example="item_view") 