"""
학습 메인 파이프라인
"""

import hydra
from omegaconf import DictConfig

from src.data.dataset import RealEstateDataset
from src.data.preprocess import preprocess_data
from src.models.lightgbm_model import LightGBMModel
from src.utils.logger import init_wandb, log_metrics, finish_wandb
from src.utils.metrics import calculate_metrics, print_metrics
from src.utils.paths import get_artifact_path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    메인 학습 함수

    Args:
        cfg: Hydra 설정
    """
    print("\n" + "="*50)
    print(f"실험 시작: {cfg.experiment.name}")
    print(f"모델: {cfg.model.name}")
    print("="*50 + "\n")

    # WandB 초기화
    init_wandb(cfg)

    # 1. 데이터 로드
    dataset = RealEstateDataset(cfg)
    train_df, test_df = dataset.load_data()

    # 2. 전처리 및 train/valid split
    X_train, X_valid, X_test, y_train, y_valid = preprocess_data(train_df, test_df, cfg)

    # 3. 모델 생성
    model = LightGBMModel(cfg)

    # 4. 모델 학습
    model.train(X_train, y_train, X_valid, y_valid)

    # 5. 평가
    # Train 평가
    train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, train_pred)
    print_metrics(train_metrics, prefix="Train")
    log_metrics(train_metrics, prefix="train/")

    # Valid 평가
    valid_pred = model.predict(X_valid)
    valid_metrics = calculate_metrics(y_valid, valid_pred)
    print_metrics(valid_metrics, prefix="Valid")
    log_metrics(valid_metrics, prefix="valid/")

    # 6. 모델 저장
    if cfg.experiment.save_model:
        model_path = get_artifact_path("models", f"{cfg.experiment.name}.pkl")
        model.save_model(model_path)

    # WandB 종료
    finish_wandb()

    print("\n" + "="*50)
    print("학습 완료!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
