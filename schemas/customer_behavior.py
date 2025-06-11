from pydantic import BaseModel, Field
from typing import Literal


class CustomerBehavior(BaseModel):
    name: str = Field(..., description="상품명", example="T-shirt")
    event: Literal[
        "item_view",
        "item_like",
        "item_add_to_cart_tap",
        "offer_make",
        "buy_start",
        "buy_comp",
    ] = Field(..., description="사용자 이벤트", example="item_view")
