from contextlib import asynccontextmanager
# --- FastAPI 애플리케이션 진입점 ---

# fastapi
from fastapi import FastAPI

# services
from services.google_shopping_service import GoogleShoppingService
from starlette.middleware.sessions import SessionMiddleware

# api: API 엔드포인트 라우터
from api.api_router import api_router


# --- 1. 애플리케이션 수명 주기 관리 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    print("--- Initializing Mercari Recommender Backend ---")
    shopping_service = GoogleShoppingService()
    app.state.shopping_service = shopping_service
    yield
    # 종료 시
    print("--- Shutting down Mercari Recommender Backend ---")
    app.state.shopping_service.quit()


# --- 2. FastAPI 앱 생성 ---
app = FastAPI(
    title="Mercari Recommender API",
    description="사용자 행동 시퀀스를 기반으로 상품을 추천하는 API입니다.",
    version="1.0.0",
    lifespan=lifespan,
)

# 세션 미들웨어 추가
app.add_middleware(SessionMiddleware, secret_key="v98fv2nyuf89v2yv2b183")

# 생성된 라우터를 FastAPI 앱에 포함시켜 '/api' 엔드포인트를 활성화합니다.
app.include_router(api_router(), prefix="/api")