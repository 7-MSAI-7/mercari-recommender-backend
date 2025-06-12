# typing
from typing import List

# fastapi
from fastapi import Request

# pytorch
import torch

# pandas
import pandas as pd

# numpy
import numpy as np

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# config
from core.config import EVENT_TO_IDX

# schemas
from schemas.customer_behavior import CustomerBehavior

# services
from services.customer_behavior_service import CustomerBehaviorService
from services.model_loader import sentence_model

# models
from models.two_tower_model import TwoTowerModel


def generate_recommendations_using_gru(
    dataframe: pd.DataFrame,
    trained_model,
    idx_to_item_id: dict,
    customer_behaviors: List[CustomerBehavior],
    top_n: int = 5,
) -> List[dict]:
    """
    사용자의 행동 시퀀스를 기반으로 상위 N개의 아이템을 추천합니다.

    Args:
        dataframe (pd.DataFrame): 전체 아이템 정보가 담긴 데이터프레임.
        trained_model (GruModel): 사전 학습된 GRU 모델.
        idx_to_item_id (dict): 모델의 출력 인덱스를 실제 item_id로 매핑하는 딕셔너리.
        customer_behaviors (List[customer_behavior]): 사용자의 행동 시퀀스 리스트.
        top_n (int, optional): 추천할 아이템의 개수. Defaults to 5.

    Returns:
        List[dict]: 추천된 아이템 목록. 각 아이템은 점수를 포함한 딕셔너리 형태입니다.
                     (예: [{'item_id': ..., 'name': ..., 'score': ...}, ...])
    """
    # 1. 모델 입력 데이터 준비
    # 시퀀스의 각 이벤트 문자열을 `EVENT_TO_IDX`를 사용해 정수 인덱스로 변환합니다.
    event_indices = [
        EVENT_TO_IDX[customer_behavior["event"]] for customer_behavior in customer_behaviors
    ]
    # 시퀀스의 각 상품 이름을 리스트로 추출합니다.
    names = [customer_behavior["name"] for customer_behavior in customer_behaviors]

    # torch.no_grad() 컨텍스트 내에서 추론을 수행하여 불필요한 그래디언트 계산을 방지합니다.
    with torch.inference_mode():
        # 2. 모델 추론
        # 상품 이름 리스트를 Sentence Transformer를 이용해 임베딩으로 변환합니다.
        # unsqueeze(0)를 통해 배치 차원(batch_size=1)을 추가합니다.
        name_embeds = sentence_model.encode(names, convert_to_tensor=True).unsqueeze(0)
        # 이벤트 인덱스 리스트를 텐서로 변환합니다.
        event_seq = torch.tensor([event_indices])

        # 준비된 데이터를 모델에 입력하여 모든 아이템에 대한 점수(logits)를 얻습니다.
        scores = trained_model(name_embeds, event_seq)

        # 3. 상위 N개 아이템 추출
        # `torch.topk`를 사용하여 가장 높은 점수를 가진 N개의 아이템의 점수와 인덱스를 찾습니다.
        top_scores, top_indices = torch.topk(scores, top_n)

        # 모델이 출력한 인덱스를 실제 item_id로 변환합니다.
        recommended_item_ids = [
            idx_to_item_id[idx.item()] for idx in top_indices.squeeze()
        ]

        # 4. 추천 결과 후처리
        # 추천된 item_id와 점수를 {item_id: score} 형태의 딕셔너리로 만듭니다.
        score_by_item_id = {
            item_id: score.item()
            for item_id, score in zip(recommended_item_ids, top_scores.squeeze())
        }

        # 원본 데이터프레임에서 추천된 아이템들의 상세 정보를 조회합니다.
        recommendations_df = dataframe[
            dataframe["item_id"].isin(recommended_item_ids)
        ]
        # 중복된 아이템 정보를 제거합니다.
        recommendations_df = recommendations_df.drop_duplicates(subset=["item_id"])
        # 결과를 API 응답 형식에 맞게 딕셔너리 리스트로 변환합니다.
        recommendations = recommendations_df[
            ["item_id", "name", "c0_name", "c1_name", "c2_name"]
        ].to_dict("records")

        # 각 추천 아이템 딕셔너리에 계산된 점수를 추가합니다.
        for recommendation in recommendations:
            recommendation["score"] = score_by_item_id[recommendation["item_id"]]

        # 최종 추천 목록을 점수(score) 기준으로 내림차순 정렬합니다.
        recommendations = sorted(
            recommendations, key=lambda x: x["score"], reverse=True
        )

        return recommendations 
    

def generate_recommendations_using_two_tower(
    customer_behaviors,
    model: TwoTowerModel,
    mappings: dict,
    top_n: int = 30,
) -> List[dict]:
    """
    Two-Tower 모델을 사용하여 특정 사용자에게 상위 N개의 아이템을 추천합니다.

    Args:
        user_id (str): 추천을 생성할 사용자의 ID.
        model (TwoTowerModel): 사전 학습된 Two-Tower 모델.
        mappings (dict): 사용자/아이템 ID와 인덱스 간의 매핑 정보.
        all_item_vectors (torch.Tensor): 사전 계산된 모든 아이템의 임베딩 벡터.
        top_n (int, optional): 추천할 아이템의 개수. Defaults to 10.

    Returns:
        List[dict]: 추천된 아이템 목록. 각 아이템은 점수를 포함한 딕셔너리 형태입니다.
    """
    with torch.inference_mode():
        num_items = len(mappings['item_categories'])
        all_item_prices_dict = mappings['item_prices']
        all_item_prices_tensor = torch.tensor([all_item_prices_dict.get(i, 0.0) for i in range(num_items)], dtype=torch.float)
        all_item_ids_tensor = torch.LongTensor(range(num_items))
        all_item_vectors = model.get_item_vector(all_item_ids_tensor, all_item_prices_tensor)

        if len(customer_behaviors) == 0:
            pass
        else:
            tfidf_vectorizer = TfidfVectorizer()
            num_items = len(mappings["item_categories"])
            all_titles = [mappings['item_titles'].get(i, "") for i in range(num_items)]
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_titles)
                
            behavior_item_vectors = []
            for customer_behavior in customer_behaviors:
                title_tfidf = tfidf_vectorizer.transform([customer_behavior["name"]])
                similarities = cosine_similarity(title_tfidf, tfidf_matrix).flatten()
                most_similar_item_idx = similarities.argmax()

                behavior_item_vector = all_item_vectors[most_similar_item_idx]
                behavior_item_vectors.append(behavior_item_vector)

            user_vector = torch.mean(torch.stack(behavior_item_vectors), dim=0)

        user_scores = torch.matmul(all_item_vectors, user_vector.T).squeeze()

        top_k_indices = torch.topk(user_scores, k=top_n).indices.tolist()

        recommendation_names = [
            mappings["item_titles"][idx] for idx in top_k_indices
        ]

        return recommendation_names