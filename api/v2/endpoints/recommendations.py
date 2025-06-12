import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from starlette.status import HTTP_202_ACCEPTED
from starlette.concurrency import run_in_threadpool

# services
from services.model_loader import initialize_trained_two_tower_model
from services.google_shopping_service import search_google_shopping
from services.customer_behavior_service import CustomerBehaviorService
from services.recommendation_service import generate_recommendations_using_two_tower

# core & schemas
from core.database import get_db, SessionLocal
import core.database as db_models
import schemas.product as schemas

trained_two_tower_model, two_tower_mappings, two_tower_df = initialize_trained_two_tower_model()


async def run_scraping_and_store_in_db(customer_behaviors: list, task_id: str):
    """
    백그라운드에서 [모델 예측 -> 스크래핑]을 수행하고 결과를 DB에 저장
    """
    try:
        # 1. 모델 예측을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
        keywords = await run_in_threadpool(
            generate_recommendations_using_two_tower,
            customer_behaviors=customer_behaviors,
            model=trained_two_tower_model,
            mappings=two_tower_mappings
        )
    except Exception as e:
        print(f"Error during model inference task {task_id}: {e}")
        db_session = SessionLocal()
        try:
            task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
            if task:
                task.status = "failed"
                task.completed_at = datetime.utcnow()
                db_session.commit()
        finally:
            db_session.close()
        return  # 추론 실패 시 작업 중단

    db_session = SessionLocal()
    try:
        # 2. 스크래핑 실행
        tasks = [search_google_shopping(keyword) for keyword in keywords[:4]]
        results = await asyncio.gather(*tasks)
        recommendation_products = [item for sublist in results for item in sublist if sublist]

        # 3. 스크래핑된 상품들을 DB에 저장
        for prod in recommendation_products:
            db_product = db_models.Product(
                task_id=task_id,
                name=prod.get("name"),
                price=prod.get("price"),
                seller=prod.get("seller"),
                image=prod.get("image"),
            )
            db_session.add(db_product)
        
        # 4. 작업 상태를 'completed'로 업데이트
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if task:
            task.status = "completed"
            task.completed_at = datetime.utcnow()
        
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        # 오류 발생 시 작업 상태를 'failed'로 업데이트 (선택적)
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if task:
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            db_session.commit()
        print(f"Error during scraping task {task_id}: {e}")
    finally:
        db_session.close()


def get_random_products(db: Session, count: int = 40) -> list:
    """
    데이터베이스의 'products' 테이블에서 무작위로 상품을 조회합니다.
    """
    return db.query(db_models.Product).order_by(func.random()).limit(count).all()


def recommendations_router():
    router = APIRouter()

    @router.post(
        "/recommendations",
        summary="새로운 추천 생성 요청 (중복 방지)",
        description="진행 중인 작업이 없을 때만 새로운 추천을 생성하고, 작업 ID를 세션에 저장합니다.",
        status_code=HTTP_202_ACCEPTED,
        response_model=schemas.TaskStatus,
    )
    async def create_recommendations(
        request: Request,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
    ):
        if "user_id" not in request.session:
            request.session["user_id"] = str(uuid.uuid4())
        user_id = request.session["user_id"]

        # DB를 직접 조회하여 이 사용자의 v2 API 작업 중 'pending' 상태인 것이 있는지 확인
        existing_pending_task = db.query(db_models.Task).filter(
            db_models.Task.user_id == user_id,
            db_models.Task.api_version == "v2",
            db_models.Task.status == "pending"
        ).first()

        if existing_pending_task:
            # 진행 중인 작업이 있으면 새로 생성하지 않고 기존 작업 정보 반환
            return existing_pending_task

        # 진행 중인 작업이 없으면 새로 생성
        customer_behaviors = CustomerBehaviorService.get_behaviors(request)[-40:]
        if len(customer_behaviors) == 0:
            item_indices = two_tower_df.sample(3)["item_idx"].tolist()
            item_names = [two_tower_mappings["item_titles"][item_idx] for item_idx in item_indices]
            customer_behaviors = [ { "name": item_name } for item_name in item_names ][:4]
        
        task_id = str(uuid.uuid4())
        new_task = db_models.Task(task_id=task_id, user_id=user_id, status="pending", api_version="v2")
        db.add(new_task)
        db.commit()
        db.refresh(new_task)
        
        # 세션에는 'v2_task' 키로 항상 최신 요청 정보를 저장
        request.session["v2_task"] = {"task_id": new_task.task_id, "status": new_task.status}
        
        background_tasks.add_task(run_scraping_and_store_in_db, customer_behaviors, task_id)

        return new_task

    @router.get(
        "/recommendations",
        summary="추천 결과 조회 (세션 또는 랜덤)",
        description="세션에 작업이 있으면 결과를 조회하고, 없으면 DB에서 랜덤 상품을 반환합니다.",
        response_model=schemas.TaskResult,
    )
    async def get_recommendations(
        request: Request,
        db: Session = Depends(get_db)
    ):
        # user_id가 없으면 먼저 생성
        if "user_id" not in request.session:
            request.session["user_id"] = str(uuid.uuid4())
            
        v2_task_info = request.session.get("v2_task")
        user_id = request.session.get("user_id")

        # 1. 세션에 작업 정보가 없는 경우 (v2_task가 없는 경우)
        if not v2_task_info or not v2_task_info.get("task_id"):
            random_products = get_random_products(db, 40)
            return {
                "task_id": "random", 
                "status": "completed", 
                "api_version": "v2", 
                "data": random_products
            }

        task_id = v2_task_info.get("task_id")
        task = db.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()

        # 2. 세션에 ID는 있지만 DB에 없는 경우 (DB 초기화 등)
        if not task:
            random_products = get_random_products(db, 40)
            return {
                "task_id": "random", 
                "status": "completed", 
                "api_version": "v2", 
                "data": random_products
            }

        # DB 상태가 세션과 다르면 세션 업데이트
        if task.status != v2_task_info.get("status"):
            request.session["v2_task"] = {"task_id": task.task_id, "status": task.status}

        response_data = {"task_id": task.task_id, "status": task.status, "api_version": task.api_version}
        
        if task.status == "completed":
            products = db.query(db_models.Product).filter(db_models.Product.task_id == task_id).all()
            
            # 3. 작업은 완료됐지만 결과 데이터가 없는 경우
            if not products:
                random_products = get_random_products(db, 40)
                response_data["data"] = random_products
            else:
                response_data["data"] = products
        elif task.status == "pending":
            # 진행 중일 경우, 이전의 마지막 성공 기록을 찾아서 반환
            previous_completed_task = db.query(db_models.Task)\
                .filter(
                    db_models.Task.user_id == user_id,
                    db_models.Task.status == 'completed'
                )\
                .order_by(db_models.Task.completed_at.desc())\
                .first()

            if previous_completed_task:
                products = db.query(db_models.Product)\
                    .filter(db_models.Product.task_id == previous_completed_task.task_id).all()
                response_data["data"] = products
            else:
                response_data["data"] = None # 이전 성공 기록이 없으면 데이터는 null
        else: # failed 등 다른 상태
             response_data["data"] = None
            
        return response_data

    return router 