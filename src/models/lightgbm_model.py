"""
LightGBM 모델
"""

import lightgbm as lgb
import joblib
from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM 회귀 모델"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.params = dict(cfg.model.params)

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        모델 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 타겟
            X_valid: 검증 데이터
            y_valid: 검증 타겟
        """
        print("\n=== LightGBM 학습 시작 ===")

        # early stopping을 위한 valid set 설정
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self.cfg.train.early_stopping.patience,
                    verbose=True
                )
            ] if self.cfg.train.early_stopping.enabled else None
        else:
            eval_set = None
            callbacks = None

        # 모델 생성 및 학습
        self.model = lgb.LGBMRegressor(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        print("=== 학습 완료 ===\n")
        return self.model

    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다")
        return self.model.predict(X)

    def save_model(self, path):
        """모델 저장"""
        joblib.dump(self.model, path)
        print(f"모델 저장 완료: {path}")

    def load_model(self, path):
        """모델 로드"""
        self.model = joblib.load(path)
        print(f"모델 로드 완료: {path}")
