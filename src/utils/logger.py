"""
WandB 실험 로깅 - 간단 버전
"""

import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig):
    """WandB 초기화"""
    if not cfg.wandb.enabled:
        print("WandB 비활성화")
        return None

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.experiment.name,
        config=config_dict,
        mode="offline" if cfg.wandb.offline else "online",
    )

    print(f"WandB 시작: {run.name}")
    return run


def log_metrics(metrics: dict, prefix: str = ""):
    """메트릭 로깅"""
    if not wandb.run:
        return

    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    wandb.log(metrics)


def finish_wandb():
    """WandB 종료"""
    if wandb.run:
        wandb.finish()
        print("WandB 종료")
