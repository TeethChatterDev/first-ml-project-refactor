"""
파일 경로 관리 및 데이터 로드 유틸리티 함수들
"""

from pathlib import Path
from typing import Union
import pandas as pd


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 경로를 반환합니다.

    Returns:
        Path: 프로젝트 루트 경로
    """
    return Path(__file__).parent.parent.parent


def get_data_path(filename: str) -> Path:
    """
    data 폴더 내의 파일 경로를 반환합니다.

    Args:
        filename: 데이터 파일 이름

    Returns:
        Path: 데이터 파일 전체 경로
    """
    return get_project_root() / "data" / filename


def get_artifact_path(artifact_type: str, filename: str = None) -> Path:
    """
    artifacts 폴더 내의 경로를 반환합니다 (models, plots, metrics 등).
    폴더가 없으면 자동으로 생성합니다.

    Args:
        artifact_type: artifact 타입 (models, plots, metrics 등)
        filename: 선택적으로 파일명 지정

    Returns:
        Path: artifact 디렉토리 또는 파일 경로
    """
    artifact_dir = get_project_root() / "artifacts" / artifact_type
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        return artifact_dir / filename
    return artifact_dir


def load_data(filename: str) -> pd.DataFrame:
    """
    data 폴더에서 CSV 파일을 로드합니다.

    Args:
        filename: CSV 파일 이름

    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    file_path = get_data_path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    return pd.read_csv(file_path, low_memory=False)


def get_latest_model(model_name: str = "lightgbm") -> Path:
    """
    가장 최근에 저장된 모델 파일을 찾습니다.

    Args:
        model_name: 모델 이름 (lightgbm 등)

    Returns:
        Path: 가장 최근 모델 파일 경로

    Raises:
        FileNotFoundError: 모델 파일이 없는 경우
    """
    models_dir = get_artifact_path("models")

    # 모델 패턴에 맞는 파일 찾기
    model_files = list(models_dir.glob(f"{model_name}_*.pkl"))

    if not model_files:
        raise FileNotFoundError(f"{models_dir}에서 {model_name} 모델을 찾을 수 없습니다.")

    # 수정 시간 기준으로 가장 최근 파일 반환
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model


def save_submission(df: pd.DataFrame, filename: str = None) -> Path:
    """
    제출용 파일을 저장합니다.

    Args:
        df: 제출용 데이터프레임
        filename: 선택적으로 파일명 지정 (없으면 타임스탬프 자동 생성)

    Returns:
        Path: 저장된 파일 경로
    """
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"

    save_path = get_artifact_path("submissions", filename)
    df.to_csv(save_path, index=False)
    print(f"제출 파일 저장 완료: {save_path}")

    return save_path
