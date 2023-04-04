import json
from typing import Iterable, Optional
from unittest.mock import Mock

import torch
from corgy import Corgy, corgyparser
from corgy.types import KeyValuePairs, SubClass
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import Annotated

_default_device_name: str
if torch.cuda.is_available():
    _default_device_name = "cuda"
else:
    _default_device_name = "cpu"
DEFAULT_DEVICE = torch.device(_default_device_name)


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
