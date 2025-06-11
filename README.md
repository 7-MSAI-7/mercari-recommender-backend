# Mercari 상품 추천 시스템 API

이 프로젝트는 사용자의 행동 시퀀스 데이터를 기반으로 다음에 관심을 가질 만한 상품을 추천하는 FastAPI 기반의 추천 시스템 API입니다.

GRU(Gated Recurrent Unit) 딥러닝 모델을 사용하여 사용자의 최근 행동 패턴(상품 조회, 좋아요, 장바구니 담기 등)을 학습하고, 이를 통해 개인화된 상품 추천 목록을 제공합니다.

## 주요 기능
- **시퀀스 기반 추천**: 사용자의 행동 순서를 고려한 GRU 모델을 통해 정교한 추천을 제공합니다.
- **실시간 추론**: FastAPI를 통해 들어온 요청을 실시간으로 처리하고 추천 결과를 반환합니다.
- **텍스트 임베딩 활용**: Sentence-Transformers를 사용하여 상품 이름(텍스트)을 의미론적 벡터로 변환, 모델의 입력으로 활용합니다.
- **Azure 연동**: Azure Machine Learning Workspace의 데이터 저장소에 있는 학습 데이터를 직접 로드합니다.
- **확장 가능한 구조**: `api`, `services`, `core`, `models` 등 역할별로 모듈화된 구조로 기능 추가 및 유지보수가 용이합니다.

---

## 디렉토리 구조
```
.
├── api
│   ├── v1
│   │   ├── endpoints
│   │   │   └── recommendations.py  # '/recommendations' 엔드포인트 정의
│   │   └──api_v1_router.py         # '/v1' 진입점 정의 
│   └── api_router.py               # '/api' API 진입점 정의
├── core
│   └── config.py                   # 프로젝트의 모든 설정값 관리
├── model_artifacts
│   └── *.pth                       # 사전 학습된 모델 가중치 파일
├── models
│   └── gru_model.py                # PyTorch GRU 모델 구조 정의
├── schemas
│   └── customer_behavior.py        # API 요청/응답 데이터 구조(Pydantic) 정의
├── services
│   ├── data_loader.py              # Azure에서 데이터 로딩 및 전처리
│   ├── model_loader.py             # 모델 및 인코더 로딩
│   ├── recommendation_service.py   # 실제 추천 로직 수행
│   └── google_shopping_service.py  # 구글 쇼핑 검색 수행
├── .env                            # 환경변수 파일
├── main.py                         # FastAPI 애플리케이션의 메인 진입점
├── requirements.txt                # Python 라이브러리 의존성 목록
└── README.md                       # 프로젝트 설명서
```

---

## 설치 및 실행 방법

### 1. 개발 환경
- Python 3.12.9
- Cuda 12.8
- dotenvx

### 2. 프로젝트 클론
```bash
git clone https://github.com/your-username/mercari-recommender-backend.git
cd mercari-recommender-backend
```

### 3. 가상환경 생성 및 활성화
가상환경을 사용하면 프로젝트별로 독립된 라이브러리 환경을 유지할 수 있습니다.
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
# install on windows
winget install dotenvx

# install on mac
brew install dotenvx/brew/dotenvx
```

### 5. 환경 변수 설정
이 프로젝트는 Azure의 리소스에 접근하기 위한 인증 정보가 필요합니다.

dotenvx를 사용하여 환경 변수를 암호화하여 관리합니다.
`.env.keys` 파일을 프로젝트 최상단(`README.md`와 같은 위치)에 생성하세요. 

### 환경 변수 암호화
```
dotenvx encrypt
```

### 환경 변수 복호화
```
dotenvx decrypt
```


### 6. 의존성 라이브러리 설치
`fastapi`와 `uvicorn`을 포함한 모든 필수 라이브러리를 설치합니다.
*참고: CUDA GPU를 사용하여 학습을 진행한다면 `torch` 관련 패키지는 사용자의 CUDA 버전에 맞게 직접 설치해야 할 수 있습니다.*
```bash
# 예시: CUDA 12.8 환경
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

### 7. FastAPI 서버 실행
`dotevx`는 `.env` 파일의 환경 변수를 자동으로 로드하고, `uvicorn`을 통해 FastAPI 애플리케이션을 실행합니다.
`--reload` 옵션은 코드 변경 시 서버를 자동으로 재시작해주는 개발용 편의 기능입니다.

```bash
dotevx run -- uvicorn main:app
```
서버가 성공적으로 실행되면 터미널에 다음과 같은 메시지가 나타납니다.
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using StatReload
...
--- Initializing Mercari Recommender Backend ---
Connecting to Azure ML Workspace...
...
--- Initialization Complete. Starting FastAPI server. ---
INFO:     Application startup complete.
```
이제 API 서버가 `http://127.0.0.1:8000` 에서 실행 중입니다.

---

## API 사용법

### 자동 생성된 API 문서
서버가 실행 중일 때 웹 브라우저에서 `http://127.0.0.1:8000/docs` 로 접속하면, FastAPI가 자동으로 생성해주는 Swagger UI 문서를 확인할 수 있습니다.
이곳에서 각 API의 상세한 명세와 사용법을 확인하고 직접 테스트해볼 수 있습니다.

### `/api/v1/recommendations` 엔드포인트
- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/api/v1/recommendations`
- **Description**: 사용자의 행동 시퀀스를 받아 추천 상품 목록을 반환합니다.

#### 요청 본문 (Request Body)
사용자의 행동 시퀀스를 JSON 배열 형태로 전달합니다. 각 요소는 `name`(상품명)과 `event`(행동 종류)를 포함해야 합니다.
- `name`    상품명 키워드: `T-Shirt`
- `event`   사용자 행동: `item_view`, `item_like`, `item_add_to_cart_tap`, `offer_make`, `buy_start`, `buy_comp`

**예시:**
```json
[
  {
    "name": "[Brand A] Classic Leather Jacket",
    "event": "item_view"
  },
  {
    "name": "Modern White Sneakers",
    "event": "item_view"
  },
  {
    "name": "White Sneakers",
    "event": "buy_start"
  }
]
```

#### `curl`을 이용한 요청 예시
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/recommendations' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "name": "[Brand A] Classic Leather Jacket",
    "event": "item_view"
  },
  {
    "name": "[Brand B] Modern White Sneakers",
    "event": "item_view"
  },
  {
    "name": "[Brand B] Modern White Sneakers",
    "event": "item_like"
  }
]'
```

#### 응답 본문 (Response Body)
추천된 상품 목록이 점수(`score`)가 높은 순으로 정렬된 JSON 배열 형태로 반환됩니다.

**예시:**
```json
[
    {
        "item_id": 12345,
        "name": "[Brand C] Vintage Denim Jeans",
        "c0_name": "Apparel",
        "c1_name": "Pants",
        "c2_name": "Jeans",
        "score": 0.875
    },
    {
        "item_id": 67890,
        "name": "[Brand D] Silk Scarf",
        "c0_name": "Accessories",
        "c1_name": "Scarves",
        "c2_name": "Silk",
        "score": 0.812
    }
]
```