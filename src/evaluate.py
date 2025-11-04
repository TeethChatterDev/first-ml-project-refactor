"""
모델 평가 함수들
"""

from src.utils.metrics import calculate_metrics, print_metrics


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    모델 평가 및 결과 출력

    Args:
        model: 학습된 모델
        X: 평가 데이터
        y: 실제 타겟
        dataset_name: 데이터셋 이름 (출력용)

    Returns:
        dict: 평가 메트릭
    """
    # 예측
    predictions = model.predict(X)

    # 메트릭 계산
    metrics = calculate_metrics(y, predictions)

    # 결과 출력
    print_metrics(metrics, prefix=dataset_name)

    return metrics


def compare_predictions(y_true, predictions_dict):
    """
    여러 모델의 예측 결과 비교

    Args:
        y_true: 실제 값
        predictions_dict: {모델명: 예측값} 딕셔너리

    Returns:
        dict: {모델명: 메트릭} 딕셔너리
    """
    results = {}

    print("\n=== 모델 비교 ===")
    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_pred)
        results[model_name] = metrics
        print_metrics(metrics, prefix=model_name)

    # 최고 성능 모델 찾기
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\n최고 성능 모델: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")

    return results
