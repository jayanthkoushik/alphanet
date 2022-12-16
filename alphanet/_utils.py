import json
import logging
import shutil
from contextlib import contextmanager, ExitStack
from functools import cached_property
from typing import Any, Dict, Iterable, Literal, Optional, Tuple
from unittest.mock import Mock

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from corgy import Corgy, corgyparser
from corgy.types import KeyValuePairs, OutputBinFile, OutputDirectory, SubClass
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


CUD_PALETTE = [  # color universal design palette
    "#000000",
    "#e69f00",
    "#56b4e9",
    "#009e73",
    "#f0e442",
    "#0072b2",
    "#d55e00",
    "#cc79a7",
]


_BASE_RC = {
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "ytick.direction": "out",
    "xtick.direction": "out",
    "ytick.major.size": 0,
    "scatter.marker": ".",
    "figure.constrained_layout.use": True,
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[default]{sourcesanspro}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{microtype}",
            r"\usepackage{mathtools}",
            r"\usepackage{amssymb}",
            r"\usepackage{bm}",
        ]
    ),
}


class PlottingConfig(Corgy, corgy_make_slots=False):
    """Configuration settings for matplotlib and seaborn.

    This class can be used standalone, or as a context manager. Example::

        plt_cfg = PlottingConfig(context="talk")
        with plt_cfg:
            ...  # plot using settings
        # Original settings restored.
        plt_cfg.config()  # permanently apply settings
    """

    context: Annotated[
        Literal["paper", "poster", "notebook", "talk"], "seaborn plotting context"
    ] = "paper"
    font: Annotated[Optional[str], "default font family override"] = None
    backend: Annotated[
        Optional[str],
        "matplotlib backend (if not specified, will use 'pgf' if LaTeX is available "
        "and matplotlib default otherwise)",
    ] = None

    DEFAULT_ASPECT_RATIO = 1.4
    _default_width_per_context = {
        "paper": 3.5,
        "notebook": 4.375,
        "poster": 8.75,
        "talk": 6.5625,
    }

    @cached_property
    def _can_use_latex(self) -> bool:
        """Check if programs needed for 'pgf' are installed and available on PATH."""
        for _prog in ("latex", "dvipng", "gs", "lualatex"):
            if shutil.which(_prog) is None:
                logging.warning(
                    "plotting with latex disabled: '%s' executable not found", _prog
                )
                return False
        return True

    def config(self):
        """Apply configurations globally."""
        _mpl_rc = _BASE_RC.copy()
        if self.backend is not None:
            _mpl_rc["backend"] = self.backend
        elif self._can_use_latex:
            _mpl_rc["backend"] = "pgf"
        if self.font is not None:
            _mpl_rc["font.family"] = [self.font]
            _mpl_rc["pgf.rcfonts"] = True
        _default_width = self._default_width_per_context[self.context]
        _default_height = _default_width / self.DEFAULT_ASPECT_RATIO
        _mpl_rc["figure.figsize"] = (_default_width, _default_height)
        sns.set_theme(
            context=self.context, style="ticks", palette=CUD_PALETTE, rc=_mpl_rc
        )

    def __enter__(self) -> None:
        self._rc_orig = mpl.rcParams.copy()
        self.config()

    def __exit__(self, exc_type, exc_val, exc_tb):
        mpl.rcParams.update(self._rc_orig)
        if exc_type is not None:
            return


class Plot(Corgy, corgy_make_slots=False):
    """Wrapper around a single plot, possibly with multiple axes.

    Objects of this class should be used in a `with` statement through the
    `open` method, which returns a new figure and axes. Leaving the context will
    close the figure, and save it to a file if provided. Example::

        plot = Plot(file="plot.png")
        with plot.open() as (fig, ax):
            ...  # plot using `fig` and `ax`
        # Figure saved to 'plot.png'.

    """

    file: Annotated[
        Optional[OutputBinFile],
        "file to save plot to (plot will not be saved if unspecified)",
    ] = None
    width: Annotated[
        Optional[float],
        "plot width in inches (if unspecified, will use value from rcParams)",
    ] = None
    aspect: Annotated[
        Optional[float],
        "plot aspect ratio, 'width/height', as a single number, or in the form "
        "width:height (if unspecified, value will be determined by default height in "
        "rcParams)",
    ] = None

    @corgyparser("aspect", metavar="float[:float]")
    @staticmethod
    def _parse_aspect(s: str) -> Optional[float]:
        if not s:
            return None
        _s_parts = s.split(":")
        if len(_s_parts) == 1:
            return float(_s_parts[0])
        if len(_s_parts) == 2:
            return float(_s_parts[0]) / float(_s_parts[1])
        raise ValueError("expected one or two values")

    def get_size(self) -> Tuple[float, float]:
        _rc_size = mpl.rcParams["figure.figsize"]
        _plot_width = self.width if self.width is not None else _rc_size[0]
        _plot_aspect = (
            self.aspect if self.aspect is not None else (_rc_size[0] / _rc_size[1])
        )
        _plot_height = _plot_width / _plot_aspect
        return (_plot_width, _plot_height)

    @contextmanager
    def open(self, **kwargs):
        """Context manager for a (figure, axes) pair.

        Figure creation can be customized by keyword arguments to the method,
        which are passed to `plt.subplots`. Example::

            plot = Plot(file="plot.png")
            with plot.open(nrows=2, ncols=1) as (fig, (ax1, ax2)):
                ... # plot with 2 rows and 1 column.
        """
        _fig, _ax = plt.subplots(**kwargs)

        if self.width is not None or self.aspect is not None:
            _fig.set_size_inches(self.get_size())

        try:
            yield (_fig, _ax)
        finally:
            if self.file is not None:
                _fig.savefig(self.file)
                self.file.close()
            plt.close(_fig)


class Plots(Corgy, corgy_make_slots=False):
    """Wrapper around a collection of plots.

    This class allows creating multiple plots, all saved to a directory. New
    plots are created by calling the `new` method, which returns a new `Plot`
    instance. Example::

        plots = Plots(dir="plots", ext="png")
        with plots.new("p1").open() as (fig, ax):
            ...  # plot using `fig` and `ax`
        # Figure saved to 'plots/p1.png'.
    """

    directory: Annotated[
        Optional[OutputDirectory],
        "directory to save plots to (plots will not be saved if unspecified)",
    ] = None
    width: Annotated[
        Optional[float],
        "default plot width in inches (if unspecified, will use value from rcParams)",
    ] = None
    aspect: Annotated[
        Optional[float],
        "default plot aspect ratio, 'width/height', as a single number, or a pair "
        "(if unspecified, value will be determined by default height in rcParams)",
    ] = None
    ext: Annotated[str, "extension for plot save files (including dot)"] = ".pdf"

    def new(
        self,
        name: Optional[str] = None,
        width: Optional[float] = None,
        aspect: Optional[float] = None,
    ) -> Plot:
        """Get a new `Plot` instance.

        Args:
            name: Filename (without extension) for the plot. The plot will not
                be saved if `None`.
            width: Value to override `Plots.width`.
            aspect: Value to override `Plots.aspect`.

        Returns:
            A new `Plot` instance.
        """
        if self.directory is not None and name is not None:
            plotfile = OutputBinFile(self.directory / f"{name}{self.ext}")
        else:
            plotfile = None
        width = width or self.width
        aspect = aspect or self.aspect
        return Plot(file=plotfile, width=width, aspect=aspect)


class ContextPlot(Plot, PlottingConfig):
    """Class combining `Plot` and `PlottingConfig`.

    This class combines the properties of `Plot` and `PlottingConfig` into a
    single class. Entering the `open` method context will configure matplotlib,
    and create a new figure and axes. Leaving the context will close the figure,
    and restore the original matplotlib configuration. Example::

        plot = ContextPlot(file="plot.png", context="talk")
        with plot.open() as (fig, ax):
            ...  # plot using `fig` and `ax`
        # Figure saved to 'plot.png'. Original matplotlib settings restored.
    """

    @contextmanager
    def open(self, **kwargs):
        stack = ExitStack()
        try:
            stack.enter_context(self)
            yield stack.enter_context(super().open(**kwargs))
        finally:
            stack.close()
