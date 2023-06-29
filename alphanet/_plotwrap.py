import logging
import math
import pickle
import shutil
from contextlib import contextmanager, ExitStack
from functools import cached_property
from pathlib import Path
from typing import cast, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from corgy import Corgy, corgyparser
from corgy.types import OutputBinFile
from typing_extensions import Annotated, Literal

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

_CUD_INVERSE = [
    "#ffffff",
    # lighter
    # "#4780ff",
    # "#ba6f45",
    # "#ff81a3",
    # "#3f49ca",
    # "#ffa471",
    # "#55b4ff",
    # "#5c9e79",
    # normal
    # "#1960ff",
    # "#a94b16",
    # "#ff618c",
    # "#0f1bbd",
    # "#ff8d4d",
    # "#2aa1ff",
    # "#338658",
    # darker
    "#144DCC",
    "#873C12",
    "#CC4E70",
    "#0C1697",
    "#CC713E",
    "#2281CC",
    "#296B46",
]

_BASE_RC = {
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "lines.color": "C0",
    "ytick.direction": "out",
    "xtick.direction": "out",
    "ytick.major.size": 0,
    "scatter.marker": ".",
    "legend.handletextpad": 0.25,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "standard",
    "savefig.edgecolor": "auto",
    "savefig.facecolor": "auto",
    "svg.fonttype": "none",
    "mathtext.bf": "bf",
    "mathtext.cal": "cursive",
    "mathtext.it": "it",
    "mathtext.rm": "rm",
    "mathtext.sf": "sf",
    "mathtext.tt": "monospace",
    "pdf.fonttype": 42,
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attr in ["serif", "sans_serif", "monospace", "cursive", "fantasy"]:
            # Remove leading/trailing spaces from font names.
            fonts: Optional[Sequence[str]] = getattr(self, attr)
            if fonts is not None:
                setattr(self, attr, [font.strip() for font in fonts])

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
    bg: Annotated[Optional[str], "plot background color override"] = None
    fg_primary: Annotated[
        Optional[str], "plot primary foreground color override"
    ] = None
    fg_secondary: Annotated[
        Optional[str], "plot secondary foreground color override"
    ] = None
    transparent: Annotated[bool, "save with transparent background"] = False

    DEFAULT_ASPECT_RATIO = 2.0
    _scale_per_context = {"paper": 1, "notebook": 1.25, "poster": 2, "talk": 3.5}
    _font_scale_per_context = {"paper": 1, "notebook": 1.125, "poster": 1.75, "talk": 3}
    _default_half_width_per_context = {
        _context: 3.25 * _scale for _context, _scale in _scale_per_context.items()
    }
    _default_full_width_per_context = {
        _context: 6.5 * _scale for _context, _scale in _scale_per_context.items()
    }
    _default_bg_per_theme = {"dark": "#212529", "light": "#ffffff"}
    _default_fg_primary_per_theme = {"dark": "#adb5bd", "light": "#212529"}
    _default_fg_secondary_per_theme = {"dark": "#495057", "light": "#adb5bd"}

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
        _font_scale = self._font_scale_per_context[self.context]
        _font_size = 9 * _font_scale
        _small_font_size = 8 * _font_scale
        _scale = self._scale_per_context[self.context]
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

        _bg = self.bg or self._default_bg_per_theme[self.theme]
        _fg_primary = self.fg_primary or self._default_fg_primary_per_theme[self.theme]
        _fg_secondary = (
            self.fg_secondary or self._default_fg_secondary_per_theme[self.theme]
        )
        _theme_rc = {
            "axes.edgecolor": _fg_primary,
            "axes.facecolor": _bg,
            "axes.labelcolor": _fg_primary,
            "boxplot.boxprops.color": _fg_primary,
            "boxplot.capprops.color": _fg_primary,
            "boxplot.flierprops.color": _fg_primary,
            "boxplot.flierprops.markeredgecolor": _fg_primary,
            "boxplot.whiskerprops.color": _fg_primary,
            "figure.edgecolor": _bg,
            "figure.facecolor": _bg,
            "grid.color": _fg_secondary,
            "patch.edgecolor": _fg_primary,
            "text.color": _fg_primary,
            "xtick.color": _fg_primary,
            "ytick.color": _fg_primary,
            "axes.prop_cycle": mpl.cycler("color", self.palette),
            "savefig.transparent": self.transparent,
        }
        mpl.rcParams.update(_theme_rc)

    @cached_property
    def palette(self):
        _c0 = self.fg_primary or self._default_fg_primary_per_theme[self.theme]
        _palette = CUD_PALETTE if self.theme == "light" else _CUD_INVERSE
        return [_c0] + _palette[1:]

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

    xlabel: Annotated[Optional[str], "label for the x axis"] = None
    ylabel: Annotated[Optional[str], "label for the y axis"] = None

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
            if self.xlim != (None, None):
                _ax.set_xlim(*self.xlim)
            if self.ylim != (None, None):
                _ax.set_ylim(*self.ylim)
            if self.xticks is not None:
                _ax.set_xticks(self.xticks)
            if self.yticks is not None:
                _ax.set_yticks(self.yticks)
            if self.xlabel is not None:
                _ax.set_xlabel(self.xlabel)
            if self.ylabel is not None:
                _ax.set_ylabel(self.ylabel)


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

    def as_dict(self, recursive=True):  # pylint: disable=arguments-differ
        d = super().as_dict(recursive)
        if self.file is not None:
            d["file"] = str(self.file)
        if self.raw_file is not None:
            d["raw_file"] = str(self.raw_file)
        d["fig"] = self.fig
        d["ax"] = self.ax
        return d

    @classmethod
    def from_dict(cls, d):  # pylint: disable=arguments-differ
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
