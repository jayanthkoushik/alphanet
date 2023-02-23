import json
import logging
import math
import pickle
import shutil
from contextlib import contextmanager, ExitStack
from functools import cached_property
from pathlib import Path
from typing import Any, cast, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union
from unittest.mock import Mock

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from corgy import Corgy, corgyparser
from corgy.types import KeyValuePairs, OutputBinFile, SubClass
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
    "legend.handletextpad": 0.25,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "standard",
    "savefig.transparent": True,
    "mathtext.bf": "bf",
    "mathtext.cal": "cursive",
    "mathtext.it": "it",
    "mathtext.rm": "rm",
    "mathtext.sf": "sf",
    "mathtext.tt": "monospace",
    "pgf.rcfonts": True,
    "pgf.preamble": "\n".join(
        [
            r"\usepackage[default]{sourcesanspro}",
            r"\usepackage{cmbright}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{microtype}",
            r"\usepackage{mathtools}",
            r"\usepackage{amssymb}",
            r"\usepackage{bm}",
        ]
    ),
}

_DARK_THEME_BG = "#212529"
_DARK_THEME_FG = "#adb5bd"
_DARK_THEME_RC = {
    "axes.edgecolor": _DARK_THEME_FG,
    "axes.facecolor": _DARK_THEME_BG,
    "axes.labelcolor": _DARK_THEME_FG,
    "boxplot.boxprops.color": _DARK_THEME_FG,
    "boxplot.capprops.color": _DARK_THEME_FG,
    "boxplot.flierprops.color": _DARK_THEME_FG,
    "boxplot.flierprops.markeredgecolor": _DARK_THEME_FG,
    "boxplot.whiskerprops.color": _DARK_THEME_FG,
    "figure.edgecolor": _DARK_THEME_BG,
    "figure.facecolor": _DARK_THEME_BG,
    "grid.color": _DARK_THEME_FG,
    "lines.color": _DARK_THEME_FG,
    "patch.edgecolor": _DARK_THEME_FG,
    "savefig.edgecolor": _DARK_THEME_BG,
    "savefig.facecolor": _DARK_THEME_BG,
    "text.color": _DARK_THEME_FG,
    "xtick.color": _DARK_THEME_FG,
    "ytick.color": _DARK_THEME_FG,
}


class PlotFont(Corgy):
    default: Annotated[Optional[str], "default font family override"] = None
    serif: Annotated[Optional[Sequence[str]], "serif font family override"] = None
    sans_serif: Annotated[
        Optional[Sequence[str]], "sans-serif font family override"
    ] = None
    monospace: Annotated[
        Optional[Sequence[str]], "monospace font family override"
    ] = None
    cursive: Annotated[Optional[Sequence[str]], "cursive font family override"] = None
    fantasy: Annotated[Optional[Sequence[str]], "fantasy font family override"] = None
    math: Annotated[
        Optional[Literal["dejavusans", "dejavuserif", "cm", "stix", "stixsans"]],
        "math font override",
    ] = None

    def config(self, rc):
        if self.default is not None:
            rc["font.family"] = self.default
        if self.serif is not None:
            rc["font.serif"] = self.serif
        if self.sans_serif is not None:
            rc["font.sans-serif"] = self.sans_serif
        if self.monospace is not None:
            rc["font.monospace"] = self.monospace
        if self.fantasy is not None:
            rc["font.fantasy"] = self.fantasy
        if self.math is not None:
            rc["mathtext.fontset"] = self.math


class PlottingConfig(Corgy, corgy_make_slots=False):
    """Configuration settings for matplotlib and seaborn.

    This class can be used standalone, or as a context manager. Example::

        plt_cfg = PlottingConfig(context="talk")
        with plt_cfg:
            ...  # plot using settings
        # Original settings restored.
        plt_cfg.config()  # permanently apply settings
    """

    theme: Annotated[Literal["light", "dark"], "plot color palette"] = "light"
    context: Annotated[
        Literal["paper", "poster", "notebook", "talk"], "seaborn plotting context"
    ] = "paper"
    backend: Annotated[Optional[str], "matplotlib backend override"] = None
    dpi: Annotated[Optional[int], "plot dpi"] = 72
    font: Annotated[PlotFont, "font config"] = PlotFont()

    DEFAULT_ASPECT_RATIO = 2
    _scale_per_context = dict(paper=1, notebook=1.5, poster=2, talk=3.5)
    _default_half_width_per_context = {
        _context: 3.5 * _scale for _context, _scale in _scale_per_context.items()
    }
    _default_full_width_per_context = {
        _context: 6.5 * _scale for _context, _scale in _scale_per_context.items()
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
        _scale = self._scale_per_context[self.context]
        _font_size = 10 * _scale
        _small_font_size = 9 * _scale
        _lw = 1.25 * _scale
        _maj_tickw = 1 * _scale
        _min_tickw = 0.75 * _scale
        _maj_ticks = 5 * _scale
        _min_ticks = 3 * _scale

        _mpl_rc = {
            "font.size": _font_size,
            "axes.labelsize": _font_size,
            "axes.titlesize": _font_size,
            "xtick.labelsize": _small_font_size,
            "ytick.labelsize": _small_font_size,
            "legend.fontsize": _small_font_size,
            "legend.title_fontsize": _font_size,
            "axes.linewidth": _maj_tickw,
            "grid.linewidth": _min_tickw,
            "lines.linewidth": _lw,
            "lines.markersize": _maj_ticks,
            "patch.linewidth": _min_tickw,
            "xtick.major.width": _maj_tickw,
            "ytick.major.width": _maj_tickw,
            "xtick.minor.width": _min_tickw,
            "ytick.minor.width": _min_tickw,
            "xtick.major.size": _maj_ticks,
            "ytick.major.size": _maj_ticks,
            "xtick.minor.size": _min_ticks,
            "ytick.minor.size": _min_ticks,
        }
        _mpl_rc.update(_BASE_RC)

        if self.backend is not None:
            _mpl_rc["backend"] = self.backend
        self.font.config(_mpl_rc)
        _default_width = self._default_full_width_per_context[self.context]
        _default_height = _default_width / self.DEFAULT_ASPECT_RATIO
        _mpl_rc["figure.figsize"] = (_default_width, _default_height)
        _mpl_rc["figure.dpi"] = self.dpi

        sns.set_theme(style="ticks", palette=self.palette, rc=_mpl_rc)
        if self.theme == "dark":
            mpl.rcParams.update(_DARK_THEME_RC)

    @cached_property
    def palette(self):
        return CUD_PALETTE if self.theme == "light" else ["white"] + CUD_PALETTE[1:]

    def __enter__(self) -> None:
        self._rc_orig = mpl.rcParams.copy()
        self.config()

    def __exit__(self, exc_type, exc_val, exc_tb):
        mpl.rcParams.update(self._rc_orig)
        if exc_type is not None:
            return


class PlotParams(Corgy, corgy_make_slots=False):
    """Parameters for plot axes."""

    log_xscale: Annotated[bool, "whether to use log scale for the x axis"] = False
    log_yscale: Annotated[bool, "whether to use log scale for the y axis"] = False

    xlim: Annotated[
        Tuple[Optional[float], Optional[float]],
        "limits for the x axis (if a value is 'inf' or '', the corresponding limit "
        "will not be set",
    ] = (None, None)
    ylim: Annotated[
        Tuple[Optional[float], Optional[float]],
        "limits for the y axis (if a value is 'inf' or '', the corresponding limit "
        "will not be set",
    ] = (None, None)

    xticks: Annotated[
        Optional[Sequence[float]],
        "ticks for the x axis (if not specified, ticks will not be modified)",
    ] = None
    yticks: Annotated[
        Optional[Sequence[float]],
        "ticks for the y axis (if not specified, ticks will not be modified)",
    ] = None

    @corgyparser("xlim", "ylim", metavar="float")
    @staticmethod
    def _parse_lim(s: str) -> Optional[float]:
        if not s or math.isinf((_f := float(s))):
            return None
        return _f

    def set_params(self, axs):
        """Set plot parameters to the given axes."""
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        else:
            axs = [axs]
        for _ax in axs:
            if self.log_xscale:
                _ax.set_xscale("log")
            if self.log_yscale:
                _ax.set_yscale("log")
            _ax.set_xlim(*self.xlim)
            _ax.set_ylim(*self.ylim)
            if self.xticks is not None:
                _ax.set_xticks(self.xticks)
            if self.yticks is not None:
                _ax.set_yticks(self.yticks)


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
    raw_file: Annotated[
        Optional[OutputBinFile], "file to save the plot object as a pickle"
    ] = None
    width: Annotated[
        Optional[float],
        "plot width in inches (if unspecified, will use value from rcParams)",
    ] = None
    aspect: Annotated[
        float,
        "plot aspect ratio, 'width/height', as a single number, or in the form "
        "width:height",
    ] = PlottingConfig.DEFAULT_ASPECT_RATIO

    def __init__(self, **args):
        super().__init__(**args)
        self.fig, self.ax = None, None

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

    def get_width(self) -> float:
        return (
            self.width if self.width is not None else mpl.rcParams["figure.figsize"][0]
        )

    def get_size(self) -> Tuple[float, float]:
        _plot_width = self.get_width()
        _plot_height = _plot_width / self.aspect
        return (_plot_width, _plot_height)

    @contextmanager
    def open(self, close_fig_on_exit: bool = True, **kwargs):
        """Context manager for a (figure, axes) pair.

        Figure creation can be customized by keyword arguments to the method,
        which are passed to `plt.subplots`. Example::

            plot = Plot(file="plot.png")
            with plot.open(nrows=2, ncols=1) as (fig, (ax1, ax2)):
                ... # plot with 2 rows and 1 column.
        """
        self.fig, self.ax = plt.subplots(**kwargs)
        self.fig.set_size_inches(self.get_size())

        try:
            yield (self.fig, self.ax)
        finally:
            if self.file is not None:
                self.fig.savefig(self.file, format=Path(self.file.name).suffix[1:])
                self.file.close()
            if self.raw_file is not None:
                pickle.dump(self.as_dict(), self.raw_file)
                self.raw_file.close()
            if close_fig_on_exit:
                plt.close(self.fig)

    def as_dict(self, recursive=True):
        d = super().as_dict(recursive)
        if self.file is not None:
            d["file"] = str(self.file)
        if self.raw_file is not None:
            d["raw_file"] = str(self.raw_file)
        d["fig"] = self.fig
        d["ax"] = self.ax
        return d

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        obj.fig = d["fig"]
        obj.ax = d["ax"]
        return obj


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

    class _widthType:
        __metavar__ = "'half'/'full'/float"

        value: Union[Literal["half", "full"], float]

        def __init__(self, s: str):
            if s in ("half", "full"):
                self.value = cast(Literal["half", "full"], s)
            else:
                self.value = float(s)

        def __str__(self):
            return str(self.value)

    width: Annotated[
        _widthType,
        "plot width in inches, or special values 'half' and 'full' which "
        "are default values based on the context",
    ] = _widthType(
        "half"
    )  # type: ignore

    def get_width(self) -> float:
        if self.width.value == "half":
            return self._default_half_width_per_context[self.context]
        if self.width.value == "full":
            return self._default_full_width_per_context[self.context]
        return self.width.value

    @contextmanager
    def open(self, close_fig_on_exit: bool = True, **kwargs):
        stack = ExitStack()
        try:
            stack.enter_context(self)
            yield stack.enter_context(super().open(close_fig_on_exit, **kwargs))
        finally:
            stack.close()
