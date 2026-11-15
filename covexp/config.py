from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # 경로
    pubtabnet_dir: Path = Path("data/pubtabnet")   # PubTabNet_2.0.0.jsonl, train/, val/
    work_dir: Path = Path("work")
    # 데이터 규모 (대상 모델 추론 수 = n_docs × (1 + 변형 9종 × 강도 3단계) = n_docs × 28)
    n_profile: int = 500        # train 분할: 뉴런 활성 범위 프로파일링용
    n_docs: int = 100           # val 분할: 테스트 원본 수
    severities: tuple = (1, 2, 3)
    seed: int = 42
    # 커버리지
    nc_thresholds: tuple = (0.0, 0.25, 0.5, 0.75)
    kmnc_ks: tuple = (10, 100)
    # 선별 실험
    set_sizes: tuple = (50, 100, 200, 500)
    n_reps: int = 30            # 반복 횟수 (반복마다 다른 부분 풀 사용)
    subpool_ratio: float = 0.7
    rq1_n: int = 100            # RQ1: 고정 집합 크기
    rq1_sets: int = 300         # RQ1: 무작위 집합 개수
    # 결함 오라클
    teds_abs: float = 0.80      # 결함 기준: 정답 HTML 대비 TEDS < teds_abs (정답 기반 단일 기준)
    # 대리 모델
    model_name: str = "microsoft/table-transformer-structure-recognition-v1.1-all"
    det_threshold: float = 0.5
    max_side: int = 1000
