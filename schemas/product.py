from pydantic import BaseModel
from typing import List, Optional

class ProductBase(BaseModel):
    name: str
    price: str
    seller: str
    image: str

class Product(ProductBase):
    id: int
    task_id: str

    class Config:
        orm_mode = True # SQLAlchemy 모델과 호환되도록 설정

class TaskStatus(BaseModel):
    task_id: str
    status: str
    api_version: Optional[str] = None

class TaskResult(TaskStatus):
    data: Optional[List[Product]] = None 