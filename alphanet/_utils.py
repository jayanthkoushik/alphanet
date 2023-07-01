import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import MulticlassAccuracy
from tqdm import trange

from ._dataset import SplitLTDataGroup, SplitLTDataset
from ._pt import DEFAULT_DEVICE, TBLogs

_TEST_DATA_CACHE: Dict[str, SplitLTDataGroup] = {}


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


def get_topk_acc(train_res, k: int, batch_size: int = 1024):
    _alphanet_classifier = train_res.load_best_alphanet_classifier().to(DEFAULT_DEVICE)
    _dataset = SplitLTDataset(train_res.train_data_info.dataset_name)
    try:
        _test_datagrp = _TEST_DATA_CACHE[str(_dataset)]
    except KeyError:
        _test_datagrp = _dataset.load_data("test")
        _TEST_DATA_CACHE[str(_dataset)] = _test_datagrp
    _test_dataset = TensorDataset(
        _test_datagrp.feat__mat, torch.tensor(_test_datagrp.label__seq)
    )
    _data_loader = DataLoader(_test_dataset, batch_size)

    _topk_metric = MulticlassAccuracy(k=k, device=DEFAULT_DEVICE)
    _topk_metric__per__split = {
        _split: MulticlassAccuracy(k=k, device=DEFAULT_DEVICE)
        for _split in ["many", "medium", "few"]
    }

    _pbar = trange(
        len(_test_dataset),
        desc=f"Computing top-{k} accuracy",
        unit="sample",
        leave=False,
    )
    for _X_batch, _y_batch in _data_loader:
        _X_batch, _y_batch = _X_batch.to(DEFAULT_DEVICE), _y_batch.to(DEFAULT_DEVICE)
        with torch.no_grad():
            _yhat_batch = _alphanet_classifier(_X_batch)
        _topk_metric.update(_yhat_batch, _y_batch)
        for _yhat_i, _yi in zip(_yhat_batch, _y_batch):
            for (
                _split,
                _split_class__set,
            ) in train_res.train_data_info.class__set__per__split.items():
                if int(_yi) in _split_class__set:
                    break
            else:
                raise AssertionError
            _topk_metric__per__split[_split].update(
                _yhat_i.unsqueeze(0), _yi.unsqueeze(0)
            )
        _pbar.update(len(_X_batch))
    _pbar.close()

    _res_dict = {"Overall": float(_topk_metric.compute())}
    for _split, _split_topk_metric in _topk_metric__per__split.items():
        _res_dict[_split.title()] = float(_split_topk_metric.compute())
    return _res_dict
