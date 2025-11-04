"""
평가 지표 계산 함수들
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    회귀 모델 평가 지표 계산

    Args:
        y_true: 실제 값
        y_pred: 예측 값

    Returns:
        dict: 평가 지표 딕셔너리 (mae, rmse, r2)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """
    평가 지표 출력

    Args:
        metrics: 평가 지표 딕셔너리
        prefix: 출력 접두사 (예: "Train", "Val")
    """
    prefix_str = f"[{prefix}] " if prefix else ""
    print(f"\n{prefix_str}평가 지표:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
