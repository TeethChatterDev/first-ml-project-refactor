
---
# 🧠 # first-ml-project-refactor (Refactored)

이 프로젝트는 제가 처음으로 진행했던 머신러닝 프로젝트를 **구조적이고 재현 가능한 형태로 리팩터링**한 버전입니다.  
처음 프로젝트에서는 코드 구조나 버전 관리가 부족했지만, 이번에는 개선 과정을 통해 성장한 점을 담았습니다.



---
## 🤖 AI 활용 방식
이번 프로젝트에서는 AI를 **개발 파트너**처럼 활용했습니다.  
제가 직접 설계 방향과 구조를 기획한 뒤, AI에게 코드 리뷰와 개선 제안을 받아  
보다 효율적이고 일관된 구조로 다듬었습니다.  
또한, 반복되는 코드나 설정 부분은 AI의 피드백을 통해 자동화하고,  
리팩터링 과정에서 코드 가독성과 유지보수성을 강화했습니다.


---

## 📌 원본 프로젝트
- [Original Group Repo](https://github.com/AIBootcamp16/upstage-ml-regression-ml_7)
  - 진행 시기: 2024.09.01 ~ 2024.09.11  
  - 역할:
    - 데이터 수집 및 전처리
    - 피처 엔지니어링 및 모델 설계
    - 모델 학습
    - 모델 평가 및 시각화

  - 한계점:
    - 코드 모듈화 미흡
    - 하드코딩된 경로
    - 재현 불가능한 실험 환경

---

## 🔧 개선 포인트
| 항목 | Before | After |
|------|---------|-------|
| 코드 구조 | Notebook 기반, 모듈화 X | src/ 폴더로 기능 분리 |
| 데이터 경로 | 하드코딩 | config.yaml로 관리 |
| 모델 학습 | 수동 파라미터 | **Hydra**를 활용해 하이퍼파라미터를 CLI에서 동적으로 변경하며 실험 가능 |
| 기록 | 수동 기록 | W&B(Weights & Biases)로 실험 결과와 성능을 추적 및 비교 |

---

## 🧩 기술 스택

**Language & Environment**
- Python (3.10+)

**Data Processing**
- Pandas, NumPy, PyYAML, tqdm

**Machine Learning**
- Scikit-learn, LightGBM

**Configuration & Workflow**
- Hydra, OmegaConf

**Experiment Tracking**
- Weights & Biases (W&B)

**Visualization**
- Matplotlib, Seaborn

**Utilities**
- Joblib (병렬 처리, 모델 저장)

**Assistant**
- ChatGPT, Claude (AI 코드 리뷰 및 구조 설계 보조)
    - 코드 품질 점검 및 개선 제안
    - 디렉토리 구조 설계 시 AI 피드백 반영
    - 반복 실험 설정 자동화 지원

---



## 📂 프로젝트 구조
```bash
first-ml-project-refactor/
│
├── README.md                         # 프로젝트 설명 문서
├── LICENSE                           # 라이선스
├── requirements.txt                  # Python 패키지 의존성
│
│
├── configs/                          # Hydra 설정 파일들
│   ├── config.yaml                   # 메인 설정
│   ├── data.yaml                     # 데이터 설정
│   ├── train.yaml                    # 학습 설정
│   ├── wandb.yaml                    # WandB 설정
│   ├── eval.yaml                     # 평가 설정
│   └── model/
│       └── lightgbm.yaml             # LightGBM 모델 설정
│
├── src/                              # 소스 코드
│   ├── train.py                      # 학습 실행 파일
│   ├── evaluate.py                   # 평가 함수
│   ├── infer.py                      # 추론 실행 파일 (루트)
│   ├── data/                         # 데이터 처리
│   │   ├── __init__.py
│   │   ├── dataset.py                # 데이터셋 로더
│   │   └── preprocess.py             # 전처리 파이프라인
│   │
│   ├── models/                       # 모델 구현
│   │   ├── __init__.py
│   │   ├── base_model.py             # 모델 인터페이스
│   │   └── lightgbm_model.py         # LightGBM 모델
│   │
│   └── utils/                        # 유틸리티
│       ├── __init__.py
│       ├── paths.py                  # 경로 관리 & 데이터 로딩
│       ├── logger.py                 # WandB 로거
│       └── metrics.py                # 평가 메트릭
│
├── data/                             # 데이터 파일
│   ├── train_sample.csv              # 샘플 학습 데이터 (테스트용)
│   ├── test_sample.csv               # 샘플 테스트 데이터
│   ├── bus_feature.csv               # 버스 정류장 데이터
│   ├── subway_feature.csv            # 지하철역 데이터
│   └── sample_submission.csv         # 제출 형식 샘플
│   
│   
│
├── artifacts/                        # 학습 결과물
│   ├── models/                       # 저장된 모델들
│   │   ├── lightgbm_20251104_111003.pkl  (최신)
│   │
│   ├── submissions/                  # 추론 결과 (제출 파일)
│   │   └── submission_20251104_111726.csv
│   │
│   ├── plots/                        # EDA 시각화 그래프
│   └── metrics/                      # 평가 메트릭 결과
│
├── notebooks/                        # Jupyter 노트북
│   └── eda.ipynb                     # 탐색적 데이터 분석
│
└── assets/                           # 리소스 파일
    └── fonts/
        └── NanumGothic-Regular.ttf   # 한글 폰트
```
---


## ⚙️ 실행 방법

### 1️⃣ 환경 설정
```bash
# 가상환경 생성 및 패키지 설치
conda create -n ml-refactor python=3.10 -y
conda activate ml-refactor
pip install -r requirements.txt

```
### 2️⃣ 학습 실행
```bash
# 기본 설정으로 학습
python -m src.train

# 하이퍼파라미터 변경 실험 (Hydra)
python -m src.train model.lr=0.01 model.max_depth=5
```
### 3️⃣ 평가 실행
```bash
python -m src.evaluate

```
### 4️⃣ 추론 실행
```bash
# 최신 학습한 모델 로드 후 추론
python -m src.infer

# 특정 모델 지정 후 추론 / 상대 경로
python -m src.infer inference.model_path=artifacts/models/lightgbm_20251104_111003.pkl

# 절대 경로
python -m src.infer inference.model_path=/root/ml_project/first-ml-project-refactor/artifacts/models/lightgbm_20251104_111003.pkl
```



> 💡 **모든 과정은 CLI 기반으로 실행 가능하며, `configs/` 내 설정 파일만 수정하면 동일한 결과를 재현할 수 있습니다.**



---
## 📚 배운 점

이번 리팩터 프로젝트를 통해 짧은 시간 안에 정처 없이 흘러갔던 이전 프로젝트를 다시 돌아볼 수 있었다.
그 과정에서 코드 구조와 모델 전체 파이프라인에 대한 이해가 훨씬 깊어졌고,
코드 재사용성과 실험 환경의 중요성을 다시 한번 느꼈다.

처음부터 이런 환경을 구축한 상태에서 모델을 학습했다면
더 좋은 결과를 얻을 수 있었을 거라는 아쉬움도 있지만,
이번 경험 덕분에 다음 프로젝트에서는 훨씬 체계적으로 접근할 자신이 생겼다.

---