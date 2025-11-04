"""
데이터 로드 및 준비
"""

import pandas as pd
from omegaconf import DictConfig
from src.utils.paths import load_data


class RealEstateDataset:
    """부동산 가격 예측 데이터셋"""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Hydra 설정
        """
        self.cfg = cfg
        self.train_df = None
        self.test_df = None
        self.target_col = cfg.data.target

    def load_data(self):
        """train, test 데이터 로드"""
        print("데이터 로딩 중...")

        self.train_df = load_data(self.cfg.data.files.train)
        self.test_df = load_data(self.cfg.data.files.test)

        print(f"Train 데이터 크기: {self.train_df.shape}")
        print(f"Test 데이터 크기: {self.test_df.shape}")

        return self.train_df, self.test_df

    def get_X_y(self, df=None, is_train=True):
        """
        X(features)와 y(target) 분리

        Args:
            df: 데이터프레임 (None이면 self.train_df 사용)
            is_train: 학습 데이터 여부

        Returns:
            tuple: (X, y) 또는 (X, None)
        """
        if df is None:
            df = self.train_df

        if is_train and self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            return X, y
        else:
            return df, None
