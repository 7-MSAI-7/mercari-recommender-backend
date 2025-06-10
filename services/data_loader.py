# azure ml sdk
from azureml.core import Workspace, Dataset, Datastore
from azureml.fsspec import AzureMachineLearningFileSystem

# system
import os

# pandas
import pandas as pd

# config
from core.config import (
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP,
    AZURE_WORKSPACE_NAME,
    AZURE_DATA_PATH,
    MIN_SEQUENCE_LENGTH,
)


def intialize_merrec_dataframe():
    """
    Azure Machine Learning 서비스에 연결하여 추천 모델 학습에 필요한 데이터를 로드하고 전처리합니다.

    - Azure Workspace에 인증합니다.
    - 지정된 경로(AZURE_DATA_PATH)에 있는 모든 Parquet 파일들을 찾습니다.
    - 각 Parquet 파일을 Pandas 데이터프레임으로 읽어와 하나로 병합합니다.
    - 최소 시퀀스 길이(MIN_SEQUENCE_LENGTH)보다 짧은 데이터를 필터링합니다.
    - 모델 학습에 사용되지 않는 불필요한 컬럼들을 제거합니다.

    Returns:
        pd.DataFrame: 전처리된 추천 시스템용 데이터프레임.
    """
    # 1. Azure Workspace 연결
    print("Connecting to Azure ML Workspace...")
    workspace = Workspace(
        AZURE_SUBSCRIPTION_ID,
        AZURE_RESOURCE_GROUP,
        AZURE_WORKSPACE_NAME,
    )
    datastore = Datastore.get(workspace, "workspaceblobstore")

    # 2. 데이터 파일 경로 설정 및 탐색
    # AzureML 파일 시스템을 통해 데이터 저장소에 접근합니다.
    file_system_path = f"azureml://subscriptions/{AZURE_SUBSCRIPTION_ID}/resourcegroups/{AZURE_RESOURCE_GROUP}/workspaces/{AZURE_WORKSPACE_NAME}/datastores/workspaceblobstore/paths/"
    fs = AzureMachineLearningFileSystem(file_system_path)
    parquet_files = fs.glob(AZURE_DATA_PATH)
    print(f"Found {len(parquet_files)} parquet files in {AZURE_DATA_PATH}.")

    # 3. 데이터 로딩 및 병합
    df_full = pd.DataFrame()
    for file in parquet_files:
        print(f"Loading file from Azure: {file}")
        # Tabular Dataset으로 Parquet 파일을 읽습니다.
        dataset = Dataset.Tabular.from_parquet_files(
            path=(datastore, file),
            validate=False,
        )
        # Pandas 데이터프레임으로 변환하여 기존 데이터프레임에 추가합니다.
        df_full = pd.concat([df_full, dataset.to_pandas_dataframe()])

    print(f"Total rows loaded: {len(df_full)}")

    # 4. 데이터 전처리
    # 최소 시퀀스 길이를 만족하는 데이터만 남깁니다.
    df_full = df_full[df_full["sequence_length"] >= MIN_SEQUENCE_LENGTH]
    print(
        f"Rows after filtering by MIN_SEQUENCE_LENGTH ({MIN_SEQUENCE_LENGTH}): {len(df_full)}"
    )

    # 불필요한 컬럼을 제거합니다.
    columns_to_drop = [
        "c0_id",
        "c1_id",
        "c2_id",
        "shipper_name",
        "shipper_id",
        "sequence_length",
        "item_condition_id",
        "item_condition_name",
        "size_id",
        "size_name",
        "brand_id",
        "brand_name",
        "color",
        "price",
        "product_id",
    ]
    df_full = df_full.drop(
        columns=columns_to_drop,
        errors="ignore",  # 해당 컬럼이 없는 경우 오류를 무시합니다.
    )
    print(f"Final columns: {df_full.columns.tolist()}")

    return df_full 