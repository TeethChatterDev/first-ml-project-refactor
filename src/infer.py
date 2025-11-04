"""
추론 및 제출 파일 생성
"""

import hydra
from omegaconf import DictConfig

from src.data.dataset import RealEstateDataset
from src.data.preprocess import preprocess_data
from src.models.lightgbm_model import LightGBMModel
from src.utils.paths import load_data, save_submission, get_latest_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    추론 메인 함수

    Args:
        cfg: Hydra 설정
    """
    print("\n" + "="*50)
    print("추론 시작")
    print("="*50 + "\n")

    # 1. 데이터 로드
    dataset = RealEstateDataset(cfg)
    train_df, test_df = dataset.load_data()

    # 2. 전처리 (train/valid/test 분리)
    _, _, X_test, _, _ = preprocess_data(train_df, test_df, cfg)

    # 3. 모델 로드
    # 특정 모델 경로가 지정되었는지 확인
    if cfg.inference.model_path is not None:
        # 사용자가 지정한 모델 경로 사용
        from pathlib import Path
        model_path = Path(cfg.inference.model_path)
        if not model_path.exists():
            print(f"오류: 지정한 모델 파일을 찾을 수 없습니다: {model_path}")
            return
        print(f"지정된 모델 로드: {model_path}")
    else:
        # 가장 최근 학습된 모델 자동 탐색
        try:
            model_path = get_latest_model(model_name=cfg.model.name)
            print(f"가장 최근 모델 발견: {model_path}")
        except FileNotFoundError as e:
            print(f"오류: {e}")
            print("먼저 train.py로 모델을 학습해주세요.")
            return

    model = LightGBMModel(cfg)
    model.load_model(model_path)

    # 4. 예측
    print("\n예측 수행 중...")
    predictions = model.predict(X_test)

    # 예측 결과를 정수로 변환 (소수점 제거)
    predictions = predictions.round().astype(int)

    # 5. Submission 파일 생성
    # sample_submission 형식 로드
    sample_submission = load_data(cfg.data.files.sample_submission)

    # 예측 결과 채우기
    submission = sample_submission.copy()
    submission[cfg.data.target] = predictions

    # 저장
    submission_path = save_submission(submission)

    print("\n" + "="*50)
    print("추론 완료!")
    print(f"제출 파일: {submission_path}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
