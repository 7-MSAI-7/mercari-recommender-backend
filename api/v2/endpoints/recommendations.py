import asyncio
import uuid
from datetime import datetime
from typing import Dict
import random

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
from core.logging_config import setup_logger
import core.database as db_models
import schemas.product as schemas

# 로거 설정
logger = setup_logger('v2_api')

trained_two_tower_model, two_tower_mappings, two_tower_df = initialize_trained_two_tower_model()

# 전역 변수로 실행 중인 작업을 관리
_running_tasks: Dict[str, asyncio.Task] = {}

async def run_v2_scraping_and_store_in_db(customer_behaviors: list, task_id: str):
    """
    백그라운드에서 [모델 예측 -> 스크래핑]을 수행하고 결과를 DB에 저장
    """
    db_session = SessionLocal()
    try:
        logger.info(f"[{task_id}] 시작: 모델 추론 및 스크래핑 작업")
        # 작업 시작 전 취소 상태 확인
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if not task or task.status == "cancelled":
            logger.info(f"[{task_id}] 작업이 취소되었거나 존재하지 않음")
            if task_id in _running_tasks:
                del _running_tasks[task_id]
            return

        logger.info(f"[{task_id}] 1단계: 모델 추론 시작")
        logger.info(f"[{task_id}] 입력 데이터: customer_behaviors={customer_behaviors}")
        # 1. 모델 예측을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
        recommendations = await run_in_threadpool(
            generate_recommendations_using_two_tower,
            customer_behaviors=customer_behaviors,
            model=trained_two_tower_model,
            mappings=two_tower_mappings
        )
        logger.info(f"[{task_id}] 모델 추론 결과: recommendations={recommendations}")

        search_keywords = [customer_behaviors[-1]["name"]] + [
            rec for rec in random.choices(recommendations, k=4)
        ]
        logger.info(f"[{task_id}] 검색 키워드 생성: search_keywords={search_keywords}")

        # 작업 취소 상태 재확인
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if not task or task.status == "cancelled":
            logger.info(f"[{task_id}] 작업이 취소됨")
            if task_id in _running_tasks:
                del _running_tasks[task_id]
            return

    except Exception as e:
        import traceback
        logger.error(f"[{task_id}] 모델 추론 중 에러 발생:")
        logger.error(f"[{task_id}] 에러 타입: {type(e).__name__}")
        logger.error(f"[{task_id}] 에러 메시지: {str(e)}")
        logger.error(f"[{task_id}] 에러 위치:\n{traceback.format_exc()}")
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if task:
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            db_session.commit()
        if task_id in _running_tasks:
            del _running_tasks[task_id]
        return

    try:
        logger.info(f"[{task_id}] 2단계: 스크래핑 시작")
        # 2. 스크래핑 실행
        tasks = [search_google_shopping(keyword) for keyword in search_keywords]
        logger.info(f"[{task_id}] 스크래핑 작업 생성 완료: {len(tasks)}개 키워드")
        results = await asyncio.gather(*tasks)
        logger.info(f"[{task_id}] 스크래핑 결과 수신: {len(results)}개 결과")

        # 작업 취소 상태 재확인
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if not task or task.status == "cancelled":
            logger.info(f"[{task_id}] 작업이 취소됨")
            if task_id in _running_tasks:
                del _running_tasks[task_id]
            return

        recommendation_products = [item for sublist in results for item in sublist if sublist]
        logger.info(f"[{task_id}] 스크래핑 결과 처리 완료: {len(recommendation_products)}개 상품")

        # 3. 스크래핑된 상품들을 DB에 저장
        logger.info(f"[{task_id}] 3단계: DB 저장 시작")
        for idx, prod in enumerate(recommendation_products, 1):
            logger.info(f"[{task_id}] 상품 {idx}/{len(recommendation_products)} 저장 시도: {prod.get('name', 'Unknown')}")
            db_product = db_models.Product(
                task_id=task_id,
                name=prod.get("name"),
                price=prod.get("price"),
                seller=prod.get("seller"),
                image=prod.get("image"),
            )
            db_session.add(db_product)
        
        # 4. 작업 상태를 'completed'로 업데이트
        logger.info(f"[{task_id}] 4단계: 작업 상태 업데이트")
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if task:
            task.status = "completed"
            task.completed_at = datetime.utcnow()
        
        db_session.commit()
        logger.info(f"[{task_id}] 작업 완료: 모든 데이터 저장됨")

    except Exception as e:
        import traceback
        logger.error(f"[{task_id}] 스크래핑/DB 저장 중 에러 발생:")
        logger.error(f"[{task_id}] 에러 타입: {type(e).__name__}")
        logger.error(f"[{task_id}] 에러 메시지: {str(e)}")
        logger.error(f"[{task_id}] 에러 위치:\n{traceback.format_exc()}")
        db_session.rollback()
        task = db_session.query(db_models.Task).filter(db_models.Task.task_id == task_id).first()
        if task:
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            db_session.commit()
    finally:
        if task_id in _running_tasks:
            del _running_tasks[task_id]
        db_session.close()
        logger.info(f"[{task_id}] 작업 종료: 리소스 정리 완료")


def get_random_products(db: Session, count: int = 40) -> list:
    """
    데이터베이스의 'products' 테이블에서 무작위로 상품을 조회합니다.
    """
    return db.query(db_models.Product).order_by(func.random()).limit(count).all()


def recommendations_router():
    router = APIRouter()

    @router.post(
        "/recommendations",
        summary="새로운 추천 생성 요청 (이전 작업 취소)",
        description="진행 중인 작업이 있으면 취소하고 새로운 추천을 생성합니다.",
        status_code=HTTP_202_ACCEPTED,
        response_model=schemas.TaskStatus,
    )
    async def create_recommendations(
        request: Request,
        db: Session = Depends(get_db),
    ):
        if "user_id" not in request.session:
            request.session["user_id"] = str(uuid.uuid4())
        user_id = request.session["user_id"]

        # 진행 중인 작업이 있으면 취소
        existing_pending_task = db.query(db_models.Task).filter(
            db_models.Task.user_id == user_id,
            db_models.Task.api_version == "v2",
            db_models.Task.status == "pending"
        ).first()

        if existing_pending_task:
            # DB 상태 업데이트
            existing_pending_task.status = "cancelled"
            existing_pending_task.completed_at = datetime.utcnow()
            db.commit()

            # 실행 중인 백그라운드 작업 취소
            if existing_pending_task.task_id in _running_tasks:
                running_task = _running_tasks[existing_pending_task.task_id]
                running_task.cancel()
                try:
                    await running_task
                except asyncio.CancelledError:
                    pass
                del _running_tasks[existing_pending_task.task_id]

        # 새로운 작업 생성
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
        
        # 백그라운드 작업 생성 및 저장
        background_task = asyncio.create_task(run_v2_scraping_and_store_in_db(customer_behaviors, task_id))
        _running_tasks[task_id] = background_task
        
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
            if task.status == "completed":
                # 작업이 성공적으로 완료된 경우에만 세션에 task_id 유지
                request.session["v2_task"] = {"task_id": task.task_id, "status": task.status}
            else:
                # 실패, 취소 등의 경우 세션에서 task_id 제거
                request.session.pop("v2_task", None)

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