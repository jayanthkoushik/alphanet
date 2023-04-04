import logging
from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from ._pt import TBLogs


def log_metrics(
    metrics: Dict[str, Any],
    acc__per__split: Dict[str, float],
    tb_logs: TBLogs,
    title: str,
    tb_n: Optional[int] = None,
):
    logging.info(
        "%s metrics: %s", title, {_k: f"{_v:.3g}" for _k, _v in metrics.items()}
    )
    logging.info(
        "%s accuracy per split: %s",
        title,
        {_k: f"{_v:.3g}" for _k, _v in acc__per__split.items()},
    )
    for _metric, _metric_val in metrics.items():
        tb_logs.writer.add_scalar(f"{title}/{_metric}", _metric_val, tb_n)
    tb_logs.writer.add_scalars(
        f"{title}/per_split_acc",
        {**acc__per__split, "overall": metrics["accuracy"]},
        tb_n,
    )


def log_alphas(
    alpha__mat: Tensor, tb_logs: TBLogs, tb_n: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    for _i in range(alpha__mat.shape[1]):
        tb_logs.writer.add_histogram(f"alphas/alpha_{_i}_hist", alpha__mat[:, _i], tb_n)

    mean_alpha__vec = alpha__mat.mean(dim=0)
    min_alpha__vec, _ = alpha__mat.min(dim=0)
    max_alpha__vec, _ = alpha__mat.max(dim=0)

    for _i, (_mean_alpha_i, _min_alpha_i, _max_alpha_i) in enumerate(
        zip(mean_alpha__vec, min_alpha__vec, max_alpha__vec)
    ):
        logging.info(
            "alpha %d: mean=%.2g, min=%.2g, max=%.2g",
            _i,
            _mean_alpha_i,
            _min_alpha_i,
            _max_alpha_i,
        )

    for _attr_name, _attr_alpha__vec in zip(
        ["mean", "min", "max"], [mean_alpha__vec, min_alpha__vec, max_alpha__vec]
    ):
        tb_logs.writer.add_scalars(
            f"alphas/{_attr_name}",
            {
                str(_i): float(_attr_alpha_i)
                for _i, _attr_alpha_i in enumerate(_attr_alpha__vec)
            },
            tb_n,
        )

    return mean_alpha__vec, min_alpha__vec, max_alpha__vec
