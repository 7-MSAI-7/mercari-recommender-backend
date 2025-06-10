# sentence transformers
from sentence_transformers import SentenceTransformer

# pytorch
import torch

# config
from core.config import (
    DEVICE,
    NAME_EMBEDDING_DIM,
    EVENT_EMBEDDING_DIM,
    GRU_HIDDEN_DIM,
    GRU_NUM_LAYERS,
    DROPOUT_RATE,
    MODEL_ARTIFACT_PATH,
    SENTENCE_MODEL_NAME,
)

# domain specific models
from models.gru_model import GruModel

# --- 모델 및 변환기 전역 로딩 ---

# Sentence Transformer 모델을 로드합니다.
# 이 모델은 상품 이름과 같은 텍스트 데이터를 고차원 벡터(임베딩)로 변환하는 데 사용됩니다.
# 애플리케이션 시작 시 한 번만 로드하여 재사용합니다.
print(f"Loading Sentence Transformer model: {SENTENCE_MODEL_NAME}...")
sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
print("Sentence Transformer model loaded.")


def initialize_trained_model(n_events, n_items):
    """
    사전 학습된 GRU 추천 모델의 가중치를 로드하고 모델을 초기화합니다.

    - `core.config`에서 정의된 하이퍼파라미터를 사용하여 GruModel을 생성합니다.
    - 지정된 경로(MODEL_ARTIFACT_PATH)에서 저장된 모델의 상태(state_dict)를 로드합니다.
    - 모델을 평가 모드(.eval())로 설정하여, 드롭아웃 등의 학습 관련 기능이 비활성화되도록 합니다.

    Args:
        n_events (int): 모델이 인식해야 할 총 이벤트의 수.
        n_items (int): 모델이 추천해야 할 총 아이템의 수.

    Returns:
        GruModel: 학습된 가중치가 로드된 GRU 모델 객체.
    """
    print("Initializing trained GRU model...")
    # 모델 구조 정의
    model_args = {
        "device": DEVICE,
        "name_embedding_dim": NAME_EMBEDDING_DIM,
        "event_embedding_dim": EVENT_EMBEDDING_DIM,
        "gru_hidden_dim": GRU_HIDDEN_DIM,
        "gru_num_layers": GRU_NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "n_events": n_events + 1, # 패딩 인덱스를 고려하여 +1
        "n_items": n_items + 1,  # 패딩 인덱스를 고려하여 +1
    }

    # 저장된 모델 가중치 로드
    print(f"Loading model state from: {MODEL_ARTIFACT_PATH}")
    old_state_dict = torch.load(
        MODEL_ARTIFACT_PATH,
        map_location=DEVICE,
    )
    # 모델 초기화 및 가중치 적용
    gru_model = GruModel(**model_args)
    gru_model.load_state_dict(old_state_dict)
    gru_model.eval()  # 평가 모드로 설정
    print("Trained GRU model initialized and ready.")

    return gru_model


def initialize_idx_to_item_id(dataframe):
    """
    모델의 출력 인덱스를 실제 item_id로 변환하기 위한 매핑 딕셔너리를 생성합니다.

    모델은 아이템을 1부터 시작하는 정수 인덱스로 인식하지만,
    실제 데이터에서는 고유한 'item_id' 문자열 또는 숫자를 사용합니다.
    이 함수는 그 둘 사이의 변환 테이블을 만듭니다.

    Args:
        dataframe (pd.DataFrame): 'item_id' 컬럼을 포함하는 데이터프레임.

    Returns:
        dict: {인덱스(int): item_id} 형태의 딕셔너리.
    """
    print("Initializing index to item_id mapping...")
    # 데이터프레임에서 고유한 item_id 목록을 가져와 (인덱스+1, item_id) 쌍으로 딕셔너리를 생성합니다.
    # 인덱스를 1부터 시작하는 이유는 보통 0을 패딩이나 알 수 없는 토큰으로 사용하기 때문입니다.
    idx_to_item_id = {
        (i + 1): item_id
        for i, item_id in enumerate(dataframe["item_id"].unique().tolist())
    }
    print(f"Created mapping for {len(idx_to_item_id)} unique items.")

    return idx_to_item_id 