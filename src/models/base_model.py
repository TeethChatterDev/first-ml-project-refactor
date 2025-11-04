"""
기본 모델 인터페이스
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """모든 모델의 기본 클래스"""

    def __init__(self, cfg):
        """
        Args:
            cfg: 모델 설정
        """
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, X):
        """예측"""
        pass

    def save_model(self, path):
        """모델 저장"""
        pass

    def load_model(self, path):
        """모델 로드"""
        pass
