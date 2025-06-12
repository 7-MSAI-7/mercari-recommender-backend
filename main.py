# --- FastAPI 애플리케이션 진입점 ---

# fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager

# api: API 엔드포인트 라우터
from api.api_router import api_router
from core.database import create_db_and_tables
from services.google_shopping_service import (
    get_google_shopping_service,
    close_google_shopping_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 애플리케이션 시작 시 실행 ---
    # 1. 데이터베이스 및 테이블 생성
    create_db_and_tables()
    print("데이터베이스 및 테이블이 성공적으로 생성되었습니다.")
    
    # 2. Google Shopping 서비스(Playwright 브라우저) 초기화
    await get_google_shopping_service()
    print("Google Shopping 서비스가 성공적으로 초기화되었습니다.")
    
    yield
    
    # --- 애플리케이션 종료 시 실행 ---
    # 1. Google Shopping 서비스(Playwright 브라우저) 종료
    await close_google_shopping_service()
    print("Google Shopping 서비스가 성공적으로 종료되었습니다.")


# --- 1. FastAPI 앱 생성 ---
app = FastAPI(
    title="Mercari Recommender API",
    description="사용자 행동 시퀀스를 기반으로 상품을 추천하는 API입니다.",
    version="1.0.0",
    lifespan=lifespan, # 라이프사이클 이벤트 핸들러 등록
)

# CORS 설정
# 중요: allow_credentials=True를 사용하려면 allow_origins에 "*"를 사용할 수 없습니다.
# 프론트엔드 애플리케이션의 정확한 출처(origin)를 명시해야 합니다.
# 예: ["http://localhost:3000", "https://your-frontend.com"]
allowed_origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 미들웨어 추가
app.add_middleware(
    SessionMiddleware,
    secret_key="v98fv2nyuf89v2yv2b183",
    # 로컬 개발 환경(HTTP)에서 교차 출처(e.g. localhost:3000 <-> localhost:8000)
    # 세션 쿠키가 정상적으로 동작하도록 'lax'로 설정합니다.
    same_site="lax", 
    https_only=False, # 로컬 HTTP 환경이므로 False로 설정
)

# 생성된 라우터를 FastAPI 앱에 포함시켜 '/api' 엔드포인트를 활성화합니다.
app.include_router(api_router(), prefix="/api")