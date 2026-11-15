# 화이트박스 커버리지인가, 출력 불일치인가

블랙박스 문서 파서의 결함 탐지를 위한 테스트 선별 기준을 비교하는 실험 코드입니다.

- **화이트박스 기준:** 오픈소스 대리 모델(Table Transformer)에서 측정한 뉴런 커버리지(NC, KMNC, NBC)
- **블랙박스 기준:** 정답 없이 대상 파서 출력만으로 계산하는 메타모픽 불일치(MR-CER, MR-TEDS), 동등 클래스 분할(EP)
- **대상 파서:** Docling (입력·출력만 사용하는 블랙박스로 취급)
- **데이터:** PubTabNet 2.0.0 검증 분할의 표 이미지 100건 × (원본 1 + 변형 9종 × 강도 3단계) = 입력 2,800개

## 결함 판정 기준

모든 결함은 **정답 HTML(Ground Truth)과 비교한 TEDS** 하나로 판정합니다. 원본 출력과의 비교는 판정에 쓰지 않고, 메타모픽 선별 기준에만 씁니다.

| 결함 유형 | 기준 |
| --- | --- |
| 표 누락 | 출력에 표가 없음 |
| 구조 오류 | 정답 대비 구조 TEDS < 0.8 |
| 내용 오류 | 구조 TEDS ≥ 0.8이고 정답 대비 전체 TEDS < 0.8 |

원본 입력이 이 기준을 통과하지 못한 문서는 분석에서 제외합니다. 기준값은 `covexp/config.py`의 `teds_abs`에서 바꿉니다.

## 폴더 구조

```
├─ run.py                메인 파이프라인 (covexp 환경)
├─ run_target.py         대상 파서 실행 (covtarget 환경)
├─ fetch_pubtabnet.py    PubTabNet에서 필요한 파일만 추출
├─ covexp/               config, data, transforms, teds, surrogate, coverage, selection, analysis
├─ scripts/run_all.bat   전체 파이프라인 일괄 실행 (Windows)
├─ requirements-main.txt
└─ requirements-target.txt
```

`data/`와 `work/`는 저장소에 포함하지 않습니다. 실행하면 생성됩니다.

## 환경 설치

의존성 충돌을 피하기 위해 conda 환경 두 개를 씁니다. 메인 환경은 transformers 4.x와 numpy 1.x가 필요하고, Docling은 별도 환경에서 실행합니다.

```bat
conda create -n covexp python=3.11 -y
conda activate covexp
python -m pip install "numpy<2" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-main.txt

conda create -n covtarget python=3.11 -y
conda activate covtarget
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-target.txt
```

실험 환경: Windows, RTX 3060 Laptop GPU(6GB), NVIDIA 드라이버 566.07.

## 데이터 준비

```bat
conda activate covexp
hf download ajimeno/PubTabNet pubtabnet.tar.gz --repo-type dataset --local-dir data\pubtabnet_raw
python fetch_pubtabnet.py data\pubtabnet_raw\pubtabnet.tar.gz
```

압축 파일(11.24GB)에서 주석 jsonl, 검증 이미지 9,115장, 학습 이미지 약 1,000장만 꺼냅니다.

## 실행

```bat
conda activate covexp
python run.py build
python run.py surrogate

conda activate covtarget
python run_target.py docling --limit 5     & REM 설치 확인용
python run_target.py docling

conda activate covexp
python run.py judge   --tgt docling
python run.py rq1     --tgt docling
python run.py select  --tgt docling
python run.py analyze --tgt docling
python run.py analyze --tgt docling --metric tg_unique
```

또는 `scripts\run_all.bat`으로 한 번에 실행합니다.

| 단계 | 산출물 (work/) | 소요 시간 (RTX 3060 Laptop) |
| --- | --- | --- |
| build | pool/ 이미지 2,800장, pool.csv, gt.json | 약 30초 |
| surrogate | acts_pool.npy, prof_lo/hi.npy, surrogate_pred.json | 약 4분 |
| run_target docling | target_pred_docling.json, target_raw_docling.jsonl | 약 63분 |
| judge | judged_docling.csv | 약 16분 |
| rq1 | rq1_docling.csv | 수 초 |
| select | results_docling.csv | 약 9분 |
| analyze | analysis_tg_defects_docling.csv, analysis_tg_unique_docling.csv | 수 초 |

## 참고

- `run_target.py smoldocling`은 예비 실험용입니다. 반복 생성(퇴화)을 감지하면 생성을 중단하고, 중단된 출력도 그대로 판정에 씁니다.
- 대상 파서가 예외로 멈춘 입력은 결함으로 세지 않고 오류로 기록합니다. 같은 명령을 다시 실행하면 오류 입력만 다시 처리합니다.
- 30회 반복은 같은 입력 풀의 부분 풀을 쓰므로 서로 독립이 아닙니다. 결과는 p값보다 평균 차이와 A12 효과 크기 중심으로 해석합니다.
