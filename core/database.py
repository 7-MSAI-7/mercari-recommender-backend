from sqlalchemy import create_engine, Column, String, Text, ForeignKey, TIMESTAMP, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

# SQLite 데이터베이스 파일 경로
DATABASE_URL = "sqlite:///./recommendations.db"

# 데이터베이스 엔진 생성
# check_same_thread: False -> FastAPI가 백그라운드 작업 등 여러 스레드에서
# 데이터베이스와 상호작용할 수 있도록 허용하는 중요한 옵션입니다.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# 데이터베이스 세션 생성을 위한 SessionLocal 클래스
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 모델 클래스가 상속받을 기본 클래스
Base = declarative_base()


# --- ORM 모델 정의 ---

class Task(Base):
    """
    백그라운드 작업의 상태를 저장하는 테이블
    """
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    status = Column(String, default="pending")
    api_version = Column(String)
    created_at = Column(TIMESTAMP, server_default=func.now())
    completed_at = Column(TIMESTAMP, nullable=True)


class Product(Base):
    """
    스크래핑된 추천 상품 정보를 저장하는 테이블
    """
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.task_id"), index=True)
    name = Column(String, index=True)
    price = Column(String)
    seller = Column(String)
    image = Column(Text) # URL은 길 수 있으므로 Text 사용


def create_db_and_tables():
    """
    데이터베이스와 정의된 모든 테이블을 생성합니다.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    API 요청 처리 중에 데이터베이스 세션을 가져오기 위한 의존성 주입 함수.
    요청이 끝나면 세션을 자동으로 닫습니다.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 