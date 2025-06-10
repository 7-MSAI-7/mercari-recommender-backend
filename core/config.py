import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Azure
# dotevx를 통해 .env 파일에서 자동으로 로드됩니다.
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
AZURE_DATA_PATH = os.getenv("AZURE_DATA_PATH")

# Device
# 모델 추론에 사용할 디바이스 설정 (현재는 CPU)
DEVICE = torch.device("cpu")

# Event to index mapping
# 사용자 행동(이벤트)을 정수 인덱스로 매핑합니다.
EVENT_TO_IDX = {
    "item_view": 1,  # 아이템 조회
    "item_like": 2,  # 좋아요
    "item_add_to_cart_tap": 3,  # 장바구니에 담기
    "offer_make": 4,  # 가격 제안
    "buy_start": 5,  # 구매 시작
    "buy_comp": 6,  # 구매 완료
}

# 데이터 전처리 설정
MIN_SEQUENCE_LENGTH = 3


# 🧠 --- GRU 모델 구조 설정 ---
# 🎁 각 상품의 이름(텍스트)을 얼마나 자세한 숫자 벡터로 표현할지 결정합니다. 숫자가 클수록 더 자세하지만 계산이 복잡해집니다.
NAME_EMBEDDING_DIM = 384
# ✨ 각 행동(예: '상품보기', '좋아요')을 얼마나 자세한 숫자 벡터로 표현할지 결정합니다.
EVENT_EMBEDDING_DIM = 6
# 🧠 GRU 모델이 한 번에 얼마나 많은 정보를 기억할지(기억 용량) 결정합니다. 높을수록 복잡한 패턴을 학습할 수 있습니다.
GRU_HIDDEN_DIM = 512
# 🏢 GRU 층을 몇 개나 쌓을지 결정합니다. 깊을수록 더 복잡한 관계를 학습할 수 있지만 과적합의 위험이 있습니다.
GRU_NUM_LAYERS = 2
# 💧 학습 시 모델의 일부 연결을 무작위로 끊어서, 모델이 학습 데이터에만 너무 의존하지 않도록(과적합 방지) 합니다.
DROPOUT_RATE = 0.5

# Model artifact path
MODEL_ARTIFACT_PATH = "model_artifacts/lastest_gru_recommender_20250610_20230501_0-4.pth"

# Sentence Transformer model
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2" 