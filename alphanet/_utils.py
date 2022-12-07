import json
import logging
from typing import Any, Dict, Iterable, Optional, Tuple
from unittest.mock import Mock

from corgy import Corgy, corgyparser
from corgy.types import KeyValuePairs, SubClass
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import Annotated


class PTOpt(Corgy):
    """Wrapper around PyTorch optimizer and learning rate scheduler.

    Usage::

        >>> opt = PTOpt(Adam, {"lr": 0.001})
        >>> net = nn.Module(...)  # some network
        >>> opt.set_weights(net.parameters())
        >>> opt.optimizer.zero_grad()
        >>> opt.optimizer.step()
    """

    __slots__ = ("optimizer", "lr_scheduler")

    class _OptimizerSubClass(SubClass[Optimizer]):
        @classmethod
        def _choices(cls):
            return tuple(
                _c
                for _c in super()._choices()
                if _c.__module__ != "torch.optim._multi_tensor"
            )

    optim_cls: Annotated[
        _OptimizerSubClass, "optimizer sub class"
    ] = _OptimizerSubClass("Adam")

    optim_params: Annotated[
        KeyValuePairs, "arguments for the optimizer"
    ] = KeyValuePairs("")

    lr_sched_cls: Annotated[
        Optional[SubClass[_LRScheduler]], "learning rate scheduler sub class"
    ] = None

    lr_sched_params: Annotated[
        KeyValuePairs, "arguments for the learning rate scheduler"
    ] = KeyValuePairs("")

    @corgyparser("optim_params")
    @corgyparser("lr_sched_params")
    @staticmethod
    def _t_params(s: str) -> KeyValuePairs:
        dic = KeyValuePairs[str, str](s)
        for k, v in dic.items():
            v = json.loads(v)
            dic[k] = v
        return dic

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = None
        self.lr_scheduler = None

    def set_weights(self, weights: Iterable[Tensor]):
        """Set weights of underlying optimizer."""
        self.optimizer = self.optim_cls(weights, **self.optim_params)
        if self.lr_sched_cls is not None:
            self.lr_scheduler = self.lr_sched_cls(  # pylint: disable=not-callable
                self.optimizer, **self.lr_sched_params
            )

    @staticmethod
    def _better_lr_sched_repr(lr_sched: _LRScheduler) -> str:
        return (
            lr_sched.__class__.__name__
            + "(\n    "
            + "\n    ".join(
                f"{k}: {v}"
                for k, v in lr_sched.state_dict().items()
                if not k.startswith("_")
            )
            + "\n)"
        )

    def __repr__(self) -> str:
        if self.optimizer is None:
            return super().__repr__()
        r = repr(self.optimizer)
        if self.lr_scheduler is not None:
            r += f"\n{self._better_lr_sched_repr(self.lr_scheduler)}"
        return r


class TBLogs:
    """TensorBoard logs type.

    Args:
        path: Path to log directory. If `None` (default), a mock instance is
            returned.

    Usage::

        tb_logs = TBLogs("tmp/tb")
        tb_logs.writer  # `SummaryWriter` instance
    """

    __metavar__ = "dir"

    def __init__(self, path: Optional[str] = None):
        self.path = path
        if path is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise RuntimeError("tensorboard not installed") from None
            self._summary_writer = SummaryWriter(path)

    @property
    def writer(self):
        if self.path is not None:
            return self._summary_writer
        return Mock()

    def __repr__(self) -> str:
        return f"TBLogs({self.path})"


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
