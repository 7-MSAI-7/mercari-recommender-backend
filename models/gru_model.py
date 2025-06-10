import torch
import torch.nn as nn


class GruModel(nn.Module):
    """
    사용자 시퀀스 데이터를 기반으로 다음 아이템을 추천하는 GRU(Gated Recurrent Unit) 모델입니다.

    이 모델은 상품 이름 임베딩과 사용자 이벤트 시퀀스를 입력으로 받아,
    다음에 올 가능성이 가장 높은 아이템을 예측하는 점수(logits)를 출력합니다.
    """

    def __init__(
        self,
        device,
        name_embedding_dim,
        event_embedding_dim,
        gru_hidden_dim,
        n_events,
        n_items,
        gru_num_layers=1,
        dropout_rate=0.3,
        **kwargs,  # 사용되지 않는 추가 인자들을 흡수합니다.
    ):
        """
        GruModel의 초기화 메서드입니다. 모델의 레이어와 파라미터를 정의합니다.

        Args:
            device (torch.device): 모델이 실행될 디바이스 (e.g., 'cpu' or 'cuda').
            name_embedding_dim (int): 상품 이름 임베딩 벡터의 차원.
            event_embedding_dim (int): 사용자 이벤트 임베딩 벡터의 차원.
            gru_hidden_dim (int): GRU 레이어의 은닉 상태 차원.
            n_events (int): 총 사용자 이벤트의 종류 수.
            n_items (int): 추천할 총 아이템의 수.
            gru_num_layers (int, optional): GRU 레이어의 수. Defaults to 1.
            dropout_rate (float, optional): 과적합 방지를 위한 드롭아웃 비율. Defaults to 0.3.
        """
        super(GruModel, self).__init__()

        self.device = device

        # --- 레이어 정의 ---

        # 이벤트 임베딩 레이어: 각 사용자 이벤트(예: '클릭', '구매')를 고정된 크기의 벡터로 변환합니다.
        # padding_idx=0은 0번 인덱스를 패딩으로 취급하여 학습에 영향을 주지 않도록 합니다.
        self.event_embedding = nn.Embedding(
            num_embeddings=n_events, embedding_dim=event_embedding_dim, padding_idx=0
        )

        # GRU의 입력으로 사용될 결합된 임베딩의 차원을 계산합니다.
        # (상품 이름 임베딩 차원 + 이벤트 임베딩 차원)
        combined_embedding_dim = name_embedding_dim + event_embedding_dim

        # GRU 레이어: 시퀀스 데이터의 시간적 패턴을 학습합니다.
        # batch_first=True는 입력 텐서의 첫 번째 차원이 배치 크기임을 의미합니다.
        self.gru = nn.GRU(
            input_size=combined_embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        # 드롭아웃 레이어: 학습 과정에서 뉴런의 일부를 무작위로 비활성화하여 과적합을 방지합니다.
        self.dropout = nn.Dropout(dropout_rate)

        # 최종 출력 레이어 (Fully Connected Layer): GRU의 출력을 받아 각 아이템에 대한 최종 예측 점수를 생성합니다.
        self.fc = nn.Linear(gru_hidden_dim, n_items)

    def forward(self, name_embeds, event_seq):
        """
        모델의 순전파 로직을 정의합니다.

        Args:
            name_embeds (torch.Tensor): 배치 단위의 상품 이름 임베딩 텐서.
                                        (batch_size, sequence_length, name_embedding_dim)
            event_seq (torch.Tensor): 배치 단위의 사용자 이벤트 시퀀스 텐서.
                                      (batch_size, sequence_length)

        Returns:
            torch.Tensor: 각 아이템에 대한 예측 점수(logits).
                          (batch_size, n_items)
        """

        # 1. 이벤트 임베딩 생성
        # event_seq의 각 이벤트 인덱스를 임베딩 벡터로 변환합니다.
        event_embeds = self.event_embedding(event_seq)

        # 2. 임베딩 결합
        # 상품 이름 임베딩과 이벤트 임베딩을 차원을 따라(dimension 2) 결합합니다.
        combined_embeds = torch.cat((name_embeds, event_embeds), dim=2)

        # 3. GRU 순전파
        # 결합된 임베딩을 GRU 레이어에 통과시켜 시퀀스의 특징을 추출합니다.
        # output: 모든 타임스텝의 은닉 상태, _: 마지막 타임스텝의 은닉 상태
        output, _ = self.gru(combined_embeds)

        # 4. 드롭아웃 적용
        # 과적합 방지를 위해 GRU 출력에 드롭아웃을 적용합니다.
        output = self.dropout(output)

        # 5. 마지막 타임스텝 출력 선택
        # 시퀀스의 모든 정보를 요약하고 있는 마지막 타임스텝의 출력만을 사용합니다.
        output = output[:, -1, :]

        # 6. 최종 예측 점수 계산
        # 선택된 출력을 선형 레이어에 통과시켜 모든 아이템에 대한 최종 점수를 얻습니다.
        logits = self.fc(output)

        return logits
