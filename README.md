# Mercari 상품 추천 시스템 API

이 프로젝트는 사용자의 행동 시퀀스 데이터를 기반으로 다음에 관심을 가질 만한 상품을 추천하는 FastAPI 기반의 추천 시스템 API입니다.

두 가지 버전의 추천 모델을 제공합니다:
- **V1**: GRU(Gated Recurrent Unit) 모델을 사용한 시퀀스 기반 추천
- **V2**: Two-Tower 모델을 사용한 임베딩 기반 추천

## 주요 기능
- **다중 모델 지원**: GRU와 Two-Tower 모델을 통해 다양한 추천 방식을 제공합니다.
- **실시간 추론**: FastAPI를 통해 들어온 요청을 실시간으로 처리하고 추천 결과를 반환합니다.
- **백그라운드 작업 관리**: 비동기 작업을 효율적으로 관리하고 필요시 취소할 수 있습니다.
- **세션 기반 사용자 관리**: SQLite 데이터베이스를 사용하여 사용자별로 독립된 세션을 통해 추천 이력을 관리합니다.
- **텍스트 임베딩 활용**: Sentence-Transformers를 사용하여 상품 이름(텍스트)을 의미론적 벡터로 변환, 모델의 입력으로 활용합니다.
- **Azure 연동**: Azure Machine Learning Workspace의 데이터 저장소에 있는 학습 데이터를 직접 로드합니다.
- **구글 쇼핑 연동**: 추천된 상품에 대해 구글 쇼핑 검색을 수행하여 실시간 가격 정보를 제공합니다.
- **확장 가능한 구조**: `api`, `services`, `core`, `models` 등 역할별로 모듈화된 구조로 기능 추가 및 유지보수가 용이합니다.

---

## 디렉토리 구조
```
.
├── api
│   ├── v1
│   │   ├── endpoints
│   │   │   ├── recommendations.py     # V1 '/recommendations' 엔드포인트
│   │   │   ├── customer_behaviors.py  # 사용자 행동 관리 엔드포인트
│   │   │   └── products.py            # V1 '/products' 엔드포인트
│   │   └── api_v1_router.py           # '/v1' 진입점 정의
│   ├── v2
│   │   ├── endpoints
│   │   │   └── recommendations.py     # V2 '/recommendations' 엔드포인트
│   │   └── api_v2_router.py           # '/v2' 진입점 정의
│   └── api_router.py                  # '/api' API 진입점 정의
├── core
│   ├── config.py                      # 프로젝트의 모든 설정값 관리
│   └── database.py                    # SQLite 데이터베이스 설정 및 모델
├── model_artifacts
│   └── *.pth                          # 사전 학습된 모델 가중치 파일
├── models
│   ├── gru_model.py                   # PyTorch GRU 모델 구조 정의
│   └── two_tower_model.py             # PyTorch Two-Tower 모델 구조 정의
├── schemas
│   ├── customer_behavior.py           # 사용자 행동 데이터 구조
│   └── product.py                     # 상품 및 작업 상태 데이터 구조
├── services
│   ├── data_loader.py                 # Azure에서 데이터 로딩 및 전처리
│   ├── model_loader.py                # 모델 및 인코더 로딩
│   ├── recommendation_service.py      # 실제 추천 로직 수행
│   ├── google_shopping_service.py     # 구글 쇼핑 검색 수행
│   └── customer_behavior_service.py   # 사용자 행동 관리 서비스
├── recommendations.db                 # SQLite 데이터베이스 파일
├── .env                               # 환경변수 파일
├── main.py                            # FastAPI 애플리케이션의 메인 진입점
├── requirements.txt                   # Python 라이브러리 의존성 목록
└── README.md                          # 프로젝트 설명서
```

---

## 설치 및 실행 방법

### 1. 개발 환경
- Python 3.12.9
- CUDA 12.8 (GPU 가속을 위한 선택사항)
- dotenvx

### 2. 프로젝트 클론
```bash
git clone https://github.com/your-username/mercari-recommender-backend.git
cd mercari-recommender-backend
```

### 3. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. dotenvx 설치
https://dotenvx.com/
```bash
# Windows
winget install dotenvx

# macOS
brew install dotenvx/brew/dotenvx
```

### 5. 환경 변수 설정
이 프로젝트는 Azure의 리소스에 접근하기 위한 인증 정보가 필요합니다.

dotenvx를 사용하여 환경 변수를 암호화하여 관리합니다.
`.env.keys` 파일을 프로젝트 최상단에 생성하세요.

### 환경 변수 관리
```bash
# 환경 변수 암호화
dotenvx encrypt

# 환경 변수 복호화
dotenvx decrypt
```

### 6. 의존성 라이브러리 설치
```bash
# CUDA 12.8 환경의 경우
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 기타 의존성 설치
pip install -r requirements.txt
```

### 7. FastAPI 서버 실행
```bash
dotenvx run -- uvicorn main:app --reload
```

서버가 성공적으로 실행되면 다음 URL에서 접근할 수 있습니다:
- API 서버: http://127.0.0.1:8000
- API 문서: http://127.0.0.1:8000/docs

---

## API 사용법

### 사용자 행동 관리 API
#### `/api/v1/customers/behaviors` 엔드포인트
- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/api/v1/customers/behaviors`
- **Description**: 사용자의 행동을 데이터베이스에 저장합니다.

#### 요청 본문 (Request Body)
```json
{
  "name": "상품명",
  "event": "item_view"  // item_view, item_like, item_add_to_cart_tap, offer_make, buy_start, buy_comp
}
```

### 상품 검색 API
#### `/api/v1/products` 엔드포인트
- **Method**: `GET`
- **URL**: `http://127.0.0.1:8000/api/v1/products`
- **Description**: 구글 쇼핑을 통해 상품을 검색하고 결과를 반환합니다.
- **Query Parameters**:
  - `q`: 검색할 상품명 (필수)
- **응답 본문 (Response Body)**
```json
[
  {
    "name": "상품명",
    "price": "가격",
    "seller": "판매자",
    "image": "이미지 URL"
  }
]
```

### 추천 API (V1/V2)
#### `/api/v{version}/recommendations` 엔드포인트
- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/api/v{version}/recommendations`
- **Description**: 사용자의 행동 시퀀스를 기반으로 추천 상품 목록을 생성합니다.
- **특징**: 
  - 진행 중인 작업이 있으면 자동으로 취소하고 새 작업을 시작합니다.
  - 작업 상태는 'pending', 'completed', 'failed', 'cancelled' 중 하나입니다.
  - 작업이 완료되면 구글 쇼핑 검색 결과를 포함한 상품 목록을 반환합니다.

#### 응답 본문 (Response Body)
```json
{
  "task_id": "작업 ID",
  "status": "pending",
  "api_version": "v1|v2"
}
```

#### 작업 상태 조회
- **Method**: `GET`
- **URL**: `http://127.0.0.1:8000/api/v{version}/recommendations`
- **Description**: 현재 작업의 상태와 결과를 조회합니다.

#### 응답 본문 (Response Body)
```json
{
  "task_id": "작업 ID",
  "status": "completed",
  "api_version": "v1|v2",
  "data": [
    {
      "name": "상품명",
      "price": "가격",
      "seller": "판매자",
      "image": "이미지 URL"
    }
  ]
}
```

### 데이터베이스 관리
- SQLite 데이터베이스(`recommendations.db`)를 사용하여 사용자 행동과 작업 상태를 영구적으로 저장합니다.
- 데이터베이스 파일은 프로젝트 루트 디렉토리에 위치합니다.
- 서버 재시작 시에도 데이터가 유지됩니다.

### 작업 상태 관리
- 모든 추천 요청은 비동기적으로 처리되며, 작업 ID를 통해 상태를 추적할 수 있습니다.
- 진행 중인 작업이 있는 상태에서 새로운 요청이 들어오면 기존 작업이 자동으로 취소됩니다.
- 작업이 취소되면 백그라운드에서 실행 중인 모든 프로세스(모델 추론, 스크래핑 등)가 중단됩니다.
- 작업 상태는 데이터베이스에 저장되어 클라이언트가 쉽게 추적할 수 있습니다.