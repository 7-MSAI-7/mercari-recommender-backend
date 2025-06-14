"""
로깅 시스템 설정 모듈

이 모듈은 프로젝트의 로깅 시스템을 설정하고 관리합니다.
각 API 버전(v1, v2)별로 독립된 로거를 제공하며,
로그는 파일과 콘솔에 동시에 출력됩니다.

주요 기능:
- 날짜별 로그 파일 생성 및 관리
- 로그 파일 크기 제한 및 로테이션
- 일관된 로그 포맷 적용
- 콘솔과 파일 동시 출력

사용 예시:
    from core.logging_config import setup_logger
    
    # v1 API용 로거 설정
    logger = setup_logger('v1_api')
    logger.info('작업 시작')
    logger.error('에러 발생', exc_info=True)
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 로그 디렉토리 생성
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 파일명 생성 (날짜별)
current_date = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"recommender_{current_date}.log")

def setup_logger(name: str) -> logging.Logger:
    """
    지정된 이름으로 로거를 설정하고 반환합니다.
    
    이 함수는 다음과 같은 특징을 가진 로거를 생성합니다:
    - 로그 레벨: INFO
    - 파일 핸들러: 날짜별 로그 파일, 최대 10MB, 최대 5개 백업
    - 콘솔 핸들러: 표준 출력으로 로그 출력
    - 포맷터: [시간] 로거이름 [로그레벨] 메시지
    
    Args:
        name (str): 로거 이름 (예: 'v1_api', 'v2_api')
                   이 이름은 로그 메시지에 포함되어 로그의 출처를 식별하는데 사용됩니다.
        
    Returns:
        logging.Logger: 설정된 로거 객체
        
    Note:
        - 동일한 이름으로 여러 번 호출해도 하나의 로거만 생성됩니다.
        - 로그 파일은 'logs' 디렉토리에 날짜별로 생성됩니다.
        - 로그 파일이 10MB를 초과하면 자동으로 새 파일이 생성됩니다.
        - 최대 5개의 백업 파일이 유지됩니다.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 있다면 추가하지 않음
    if logger.handlers:
        return logger
    
    # 파일 핸들러 설정 (최대 10MB, 최대 5개 파일 유지)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 