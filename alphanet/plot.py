import itertools
import logging
import math
from collections import defaultdict
from heapq import nlargest, nsmallest
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import ignite.contrib.handlers
import ignite.engine
import ignite.metrics
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from corgy import Corgy
from corgy.types import InputBinFile, InputDirectory, OutputTextFile
from matplotlib import (
    image as mimg,
    patches as mpatches,
    path as mpath,
    patheffects as pe,
    pyplot as plt,
)
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    MultipleLocator,
    PercentFormatter,
)
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing_extensions import Literal

from alphanet._dataset import SplitLTDataGroup, SplitLTDataset
from alphanet._plotwrap import ContextPlot, PlotParams
from alphanet._pt import DEFAULT_DEVICE
from alphanet._samplers import AllFewSampler, ClassBalancedBaseSampler
from alphanet._wordnet import get_wordnet_nns_per_imgnet_class
from alphanet.train import TrainResult

_TEST_DATA_CACHE: Dict[str, SplitLTDataGroup] = {}
_NNDIST_PER_SPLIT_CLASS_CACHE: Dict[
    Tuple[SplitLTDataset, str, int, str, str],
    Tuple[Dict[int, float], Dict[int, List[int]]],
] = {}
_BASELINE_RES_CACHE: Dict[SplitLTDataset, TrainResult] = {}


def _get_split_test_acc(res: TrainResult, split: str) -> float:
    if split.lower() == "overall":
        return res.test_metrics["accuracy"]
    return res.test_acc__per__split[split.lower()]


def _get_nn_dist_per_split_class(
    dataset: SplitLTDataset,
    metric: str,
    k: int,
    for_split: Literal["few", "base", "all"] = "base",
    against_split: Literal["few", "base", "all"] = "few",
) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    try:
        return _NNDIST_PER_SPLIT_CLASS_CACHE[
            dataset, metric, k, for_split, against_split
        ]
    except KeyError:
        pass

    logging.info(
        "generating nn dists: '%s'->'%s' (dataset='%s', metric='%s', k='%d')",
        for_split,
        against_split,
        dataset,
        metric,
        k,
    )

    _train_data = dataset.load_data("train")

    _mean_feat__vec__per__class__msplit: Dict[str, Dict[int, torch.Tensor]] = {
        "base": defaultdict(lambda: torch.zeros(_train_data.info.n_features)),
        "few": defaultdict(lambda: torch.zeros(_train_data.info.n_features)),
    }
    for _label, _feat__vec in zip(_train_data.label__seq, _train_data.feat__mat):
        _msplit = (
            "few"
            if _label in _train_data.info.class__set__per__split["few"]
            else "base"
        )
        _mean_feat__vec__per__class__msplit[_msplit][_label] += (
            _feat__vec / _train_data.info.n_imgs__per__class[_label]
        )

    _class__seq__per__msplit: Dict[str, List[int]] = {}
    _mean_feat__mat__per__msplit: Dict[str, torch.Tensor] = {}
    for _msplit in ("base", "few"):
        _class__seq, _mean_feat__mats = zip(
            *_mean_feat__vec__per__class__msplit[_msplit].items()
        )
        _class__seq__per__msplit[_msplit] = list(_class__seq)
        _mean_feat__mat__per__msplit[_msplit] = torch.stack(_mean_feat__mats)

    _class__seq__per__msplit["all"] = (
        _class__seq__per__msplit["few"] + _class__seq__per__msplit["base"]
    )
    _mean_feat__mat__per__msplit["all"] = torch.cat(
        [_mean_feat__mat__per__msplit["few"], _mean_feat__mat__per__msplit["base"]]
    )

    _n_many, _n_med, _n_few = [
        len(_train_data.info.class__set__per__split[_split])
        for _split in ("many", "medium", "few")
    ]
    _n_base = _n_many + _n_med
    _n_all = _n_base + _n_few
    _n_feats = _train_data.info.n_features
    assert len(_class__seq__per__msplit["base"]) == _n_base
    assert _mean_feat__mat__per__msplit["base"].shape == (_n_base, _n_feats)
    assert len(_class__seq__per__msplit["few"]) == _n_few
    assert _mean_feat__mat__per__msplit["few"].shape == (_n_few, _n_feats)
    assert len(_class__seq__per__msplit["all"]) == _n_all
    assert _mean_feat__mat__per__msplit["all"].shape == (_n_all, _n_feats)

    _cdist__mat: torch.Tensor
    _n_dict = {"few": _n_few, "base": _n_base, "all": _n_all}
    _n_for = _n_dict[for_split]
    _n_against = _n_dict[against_split]
    _XA = _mean_feat__mat__per__msplit[for_split]
    _XB = _mean_feat__mat__per__msplit[against_split]
    if metric == "euclidean":
        _cdist__mat = torch.cdist(_XA.unsqueeze(0), _XB.unsqueeze(0), p=2)
        _cdist__mat = _cdist__mat.squeeze(0)
    elif metric == "cosine":
        _nXA = F.normalize(_XA)
        _nXB = F.normalize(_XB)
        _cdist__mat = 1 - (_nXA @ _nXB.t())
    else:
        raise AssertionError
    assert _cdist__mat.shape == (_n_for, _n_against)

    _nndist__mat, _nnidx__mat = _cdist__mat.sort(dim=1)

    if against_split in ["all", for_split]:
        assert all(
            _class__seq__per__msplit[against_split][_nnidx__mat[_i, 0]]
            == _class__seq__per__msplit[for_split][_i]
            for _i in range(_n_for)
        )
        _nndist__mat = _nndist__mat[:, 1:]
        _nnidx__mat = _nnidx__mat[:, 1:]

    _nndist__mat, _nnidx__mat = _nndist__mat[:, :k], _nnidx__mat[:, :k]
    assert _nndist__mat.shape == _nnidx__mat.shape == (_n_for, k)

    nn_class__seq__per__class = {}
    for (
        _class,
        _class_feat__vec,
        _class_nndist__vec,
        _class_dist__vec,
        _nn_idx__vec,
    ) in zip(
        _class__seq__per__msplit[for_split],
        _mean_feat__mat__per__msplit[for_split],
        _nndist__mat,
        _cdist__mat,
        _nnidx__mat,
    ):
        _nn_class__seq = [
            _class__seq__per__msplit[against_split][int(_idx)] for _idx in _nn_idx__vec
        ]
        nn_class__seq__per__class[_class] = _nn_class__seq

        assert all(
            torch.isclose(_class_dist__vec[int(_idx)], _class_nndist__vec[_i])
            for _i, _idx in enumerate(_nn_idx__vec)
        )

        for _nn_class, _nn_class_dist in zip(_nn_class__seq, _class_nndist__vec):
            _nn_class_feat__vec = next(
                __mean_feat__vec
                for __class, __mean_feat__vec in zip(
                    _class__seq__per__msplit[against_split],
                    _mean_feat__mat__per__msplit[against_split],
                )
                if __class == _nn_class
            )
            if metric == "euclidean":
                _computed_dist = (_class_feat__vec - _nn_class_feat__vec).norm(p=2)
            else:
                _norm_class_feat__vec = F.normalize(_class_feat__vec, dim=0)
                _norm_nn_class_feat__vec = F.normalize(_nn_class_feat__vec, dim=0)
                _computed_dist = 1 - (_norm_class_feat__vec @ _norm_nn_class_feat__vec)
            assert torch.isclose(_computed_dist, _nn_class_dist, atol=1e-3, rtol=1e-5)

    nn_dist__per__class = {
        _class: _nndist__vec.mean().item()
        for _class, _nndist__vec in zip(
            _class__seq__per__msplit[for_split], _nndist__mat
        )
    }
    _NNDIST_PER_SPLIT_CLASS_CACHE[dataset, metric, k, for_split, against_split] = (
        nn_dist__per__class,
        nn_class__seq__per__class,
    )
    return (nn_dist__per__class, nn_class__seq__per__class)


def _get_test_acc_per_class(
    res: TrainResult, batch_size: int, return_preds=False
) -> Tuple[Dict[int, float], Optional[List[int]]]:
    alphanet_classifier = res.load_best_alphanet_classifier()
    alphanet_classifier = alphanet_classifier.to(DEFAULT_DEVICE).eval()
    dataset = SplitLTDataset(res.train_data_info.dataset_name)
    try:
        test_datagrp = _TEST_DATA_CACHE[res.train_data_info.dataset_name]
    except KeyError:
        test_datagrp = dataset.load_data(res.training_config.test_datagrp)
        _TEST_DATA_CACHE[res.train_data_info.dataset_name] = test_datagrp
    test_data_loader = DataLoader(
        TensorDataset(test_datagrp.feat__mat, torch.tensor(test_datagrp.label__seq)),
        batch_size,
        shuffle=False,
    )
    eval_engine = ignite.engine.create_supervised_evaluator(
        alphanet_classifier,
        {"accuracy": ignite.metrics.Accuracy()},  # for sanity check
        DEFAULT_DEVICE,
    )
    ignite.contrib.handlers.ProgressBar(
        desc="Generating test results for file", persist=False
    ).attach(eval_engine)

    @eval_engine.on(ignite.engine.Events.STARTED)
    def _(engine: ignite.engine.Engine):
        engine.state.my_y__seq = []
        engine.state.my_yhat__seq = []
        engine.state.my_accuracy__per__class = None

    @eval_engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def _(engine: ignite.engine.Engine):
        _scores__batch, _y__batch = engine.state.output  # type: ignore
        _yhat__batch = torch.argmax(_scores__batch, dim=1)
        engine.state.my_y__seq.extend(_y__batch.tolist())
        engine.state.my_yhat__seq.extend(_yhat__batch.tolist())

    @eval_engine.on(ignite.engine.Events.COMPLETED)
    def _(engine: ignite.engine.Engine):
        _correct_preds__per__class: Dict[int, int] = defaultdict(int)
        _total_preds__per__class: Dict[int, int] = defaultdict(int)

        for _y_i, _yhat_i in zip(engine.state.my_y__seq, engine.state.my_yhat__seq):
            _correct = int(_y_i == _yhat_i)
            _correct_preds__per__class[_y_i] += _correct
            _total_preds__per__class[_y_i] += 1

        engine.state.my_accuracy__per__class = {
            _class: float(
                _correct_preds__per__class[_class] / _total_preds__per__class[_class]
            )
            for _class in _correct_preds__per__class
        }

        assert all(
            _total_preds__per__class[_class] == _class_imgs
            for _class, _class_imgs in test_datagrp.info.n_imgs__per__class.items()
        )
        assert torch.isclose(
            torch.tensor([sum(_correct_preds__per__class.values())], dtype=torch.float),
            torch.tensor(
                [engine.state.metrics["accuracy"] * len(engine.state.my_y__seq)]
            ),
        )
        assert torch.isclose(
            torch.tensor([engine.state.metrics["accuracy"]]),
            torch.tensor([res.test_metrics["accuracy"]]),
            rtol=1e-3,
            atol=1e-5,
        )
        assert all(
            _l == _y for _l, _y in zip(test_datagrp.label__seq, engine.state.my_y__seq)
        )

    eval_engine.run(test_data_loader)
    _second_ret = eval_engine.state.my_yhat__seq if return_preds else None
    return eval_engine.state.my_accuracy__per__class, _second_ret


def _load_train_res(res_file: Path, batch_size: int) -> TrainResult:
    res = TrainResult.from_dict(torch.load(res_file, map_location=DEFAULT_DEVICE))
    if res.test_acc__per__class is None:
        res.test_acc__per__class, _ = _get_test_acc_per_class(res, batch_size)
        torch.save(res.as_dict(recursive=True), res_file)
    return res


def _load_baseline_res(dataset: SplitLTDataset, batch_size: int) -> TrainResult:
    try:
        return _BASELINE_RES_CACHE[dataset]
    except KeyError:
        baseline_res = _load_train_res(dataset.baseline_eval_file_path, batch_size)
        _BASELINE_RES_CACHE[dataset] = baseline_res
        return baseline_res


def _get_classes_ordered_by_size(res: TrainResult, reverse: bool = False) -> List[int]:
    _n_imgs__per__cls = res.train_data_info.n_imgs__per__class
    return list(
        sorted(_n_imgs__per__cls, key=lambda _c: _n_imgs__per__cls[_c], reverse=reverse)
    )


def _gradient_image(ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
    # From 'matplotlib.org/stable/gallery/lines_bars_and_markers/gradient_bar.html'.
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* keyword argument.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]], [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation="bicubic", vmin=0, vmax=1, **kwargs)
    return im


class BasePlotCmd(Corgy, corgy_make_slots=False):
    log_level: Literal["error", "warning", "info", "debug"] = "error"
    plot: ContextPlot

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.root.setLevel(getattr(logging, self.log_level.upper()))

    def __call__(self) -> mpl.figure.Figure:
        raise NotImplementedError

    def _save_figure(self, fig: mpl.figure.Figure, **kwargs) -> None:
        if self.plot.file is not None:
            fig.savefig(
                self.plot.file, format=Path(self.plot.file.name).suffix[1:], **kwargs
            )
            self.plot.file.close()


class _BaseMultiExpPlotCmd(Corgy, corgy_make_slots=False):
    base_res_dir: InputDirectory
    exp_sub_dirs: Tuple[str]
    exp_names: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    eval_batch_size: int = 1024
    n_boot: int = 1000

    def _train_results(self) -> Iterator[Tuple[int, str, SplitLTDataset, TrainResult]]:
        exp_names = self.exp_sub_dirs if self.exp_names is None else self.exp_names
        if len(exp_names) != len(self.exp_sub_dirs):
            raise ValueError("`exp_names` size mismatch with `exp_sub_dirs`")

        file_id = 0
        for exp_sub_dir, exp_name in zip(self.exp_sub_dirs, exp_names):
            for res_file in tqdm(
                list((self.base_res_dir / exp_sub_dir).glob(self.res_files_pattern)),
                desc=f"Loading from '{exp_sub_dir}'",
                unit="file",
            ):
                res = _load_train_res(res_file, self.eval_batch_size)
                dataset = SplitLTDataset(res.train_data_info.dataset_name)
                yield (file_id, exp_name, dataset, res)
                file_id += 1


class PlotSplitClsAccDeltaVsNNDist(_BaseMultiExpPlotCmd, BasePlotCmd):
    splits: Sequence[Literal["many", "medium", "few", "base"]]
    dist_for_rand: Literal["euclidean", "cosine"] = "euclidean"
    col_wrap: Optional[int] = None
    sharex: bool = False
    sharey: bool = False
    fit_reg: bool = True
    plot_r: bool = True
    r_group_classes: bool = False
    plot_r_loc: Tuple[float, float] = (0.75, 0.9)
    r_font_size: Literal[
        "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"
    ] = "x-small"
    dist_on_x: bool = True
    acc: Literal["baseline", "alphanet", "delta"] = "alphanet"
    legend_ncols: Optional[int] = None
    legend_loc: str = "lower center"
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, 1)
    despine: bool = True
    add_axes_guide: bool = False
    rasterize_scatter: bool = False
    plot_params: PlotParams = PlotParams()

    def __call__(self):
        if "base" in self.splits and ("many" in self.splits or "medium" in self.splits):
            raise ValueError(
                "cannot have 'base' in `splits` along with 'many'/'medium'"
            )

        if len(self.splits) != len(set(self.splits)):
            raise ValueError("cannot have duplicates in `splits`")

        df_rows = []
        for _fid, _exp_name, _dataset, _res in self._train_results():
            _class__set__per__split = _res.train_data_info.class__set__per__split
            _acc__per__class = _res.test_acc__per__class

            _metric = _res.nn_info.nn_dist
            assert _metric in ["euclidean", "cosine", "random"]
            if _metric == "random":
                if self.splits != ["few"]:
                    raise ValueError("random distance only available for 'few' split")
                _metric = self.dist_for_rand

            _f_metric = euclidean if _metric == "euclidean" else cosine
            _nn_dist__per__fclass = {}
            for (
                _fclass,
                _nn_mean_feat__mat,
            ) in _res.nn_info.nn_mean_feat__mat__per__fclass.items():
                assert _nn_mean_feat__mat.shape == (
                    _res.nn_info.n_neighbors + 1,
                    _res.train_data_info.n_features,
                )
                _fclass__vec = _nn_mean_feat__mat[0]
                _nn_dist__per__fclass[_fclass] = mean(
                    _f_metric(_fclass__vec, _nn__vec)
                    for _nn__vec in _nn_mean_feat__mat[1:]
                )
            assert set(_nn_dist__per__fclass.keys()) == _class__set__per__split["few"]

            _nn_dist__per__bclass, _ = _get_nn_dist_per_split_class(
                _dataset, _metric, _res.nn_info.n_neighbors
            )

            assert not set(_nn_dist__per__bclass).intersection(
                set(_nn_dist__per__fclass)
            )
            _nn_dist__per__class = {**_nn_dist__per__fclass, **_nn_dist__per__bclass}
            assert len(_nn_dist__per__class) == _res.train_data_info.n_classes

            _baseline_res = _load_baseline_res(_dataset, self.eval_batch_size)
            _baseline_acc__per__class = _baseline_res.test_acc__per__class

            def _get_class_split(_c):
                # pylint: disable=cell-var-from-loop
                nonlocal _class__set__per__split
                if _c in _class__set__per__split["few"]:
                    return "few"
                if "base" in self.splits:
                    return "base"
                if _c in _class__set__per__split["medium"]:
                    return "medium"
                return "many"

            for _class in _nn_dist__per__class:
                df_rows.append(
                    {
                        "FID": _fid,
                        "Experiment": _exp_name,
                        "Class": _class,
                        "Split": _get_class_split(_class),
                        "Baseline accuracy": _baseline_acc__per__class[_class],
                        "AlphaNet accuracy": _acc__per__class[_class],
                        "Test accuracy change": (
                            _acc__per__class[_class] - _baseline_acc__per__class[_class]
                        ),
                        "Mean NN distance": _nn_dist__per__class[_class],
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)
        df = df[df["Split"].isin(self.splits)]
        _palette = dict(zip(["few", "base", "many", "medium"], self.plot.palette[:4]))

        self.plot.config()
        _extra_artists = []
        if self.acc == "baseline":
            acc_field = "Baseline accuracy"
        elif self.acc == "alphanet":
            acc_field = "AlphaNet accuracy"
        else:
            acc_field = "Test accuracy change"
        x, y = "Mean NN distance", acc_field
        if not self.dist_on_x:
            x, y = (y, x)
        g = sns.lmplot(
            df,
            x=x,
            y=y,
            hue="Split",
            col="Experiment",
            palette=_palette,
            col_wrap=self.col_wrap,
            aspect=1,
            legend=False,
            scatter=False,
            fit_reg=self.fit_reg,
            n_boot=self.n_boot,
            units="Class",
            line_kws={
                "path_effects": [
                    pe.Stroke(
                        linewidth=(mpl.rcParams["lines.linewidth"] * 1.75),
                        foreground=self.plot.palette[0],
                        alpha=0.5,
                    ),
                    pe.Normal(),
                ]
            },
            facet_kws={"sharex": self.sharex, "sharey": self.sharey},
        )
        g.map_dataframe(
            sns.scatterplot,
            x=x,
            y=y,
            hue="Split",
            style="Split",
            palette=_palette,
            markers={"few": ".", "base": "*", "many": "*", "medium": "d"},
            legend=False,
            s=(mpl.rcParams["lines.markersize"] ** 2) / 2,
            alpha=0.5,
            linewidths=0,
            rasterized=self.rasterize_scatter,
        )

        def annotate_with_corr(data, **kwargs):
            _splits = data["Split"].unique()
            assert len(_splits) == 1
            _split = _splits[0]
            if self.r_group_classes:
                data = data.groupby("Class").mean().reset_index()
            _corr = data[[acc_field, "Mean NN distance"]].corr().values[0, 1]
            _ax = plt.gca()
            _corr_text = f"$r={_corr:+.2f}$"
            _corr_text = "\n" * self.splits.index(_split) + _corr_text
            _color = self.plot.palette[0] if len(self.splits) == 1 else _palette[_split]
            _text_artist = _ax.text(
                *self.plot_r_loc,
                _corr_text,
                transform=_ax.transAxes,
                fontsize=self.r_font_size,
                color=_color,
                va="top",
                ha="right",
            )
            _extra_artists.append(_text_artist)

        if self.plot_r:
            g.map_dataframe(annotate_with_corr)

        if len(df["Experiment"].unique()) == 1:
            g.set_titles("")
        else:
            g.set_titles("{col_name}")
        g.despine(
            left=self.despine, right=self.despine, top=self.despine, bottom=self.despine
        )
        for _ax in g.axes.flat:
            self.plot_params.set_params(_ax)
            _ax.grid(visible=True, axis="both", which="major")
            _ax.tick_params(axis="both", which="major", length=0)
            if self.add_axes_guide:
                _ax.set_xlabel("")
                _ax.set_ylabel("")

        if self.add_axes_guide:
            _ax = g.figure.add_subplot(g._nrow, g._ncol, g._nrow * g._ncol)
            _ax.set_aspect("equal")
            for _del in [(0.9, 0), (0, 0.9)]:
                _ax.arrow(
                    0.05,
                    0.05,
                    *_del,
                    lw=mpl.rcParams["axes.linewidth"],
                    length_includes_head=True,
                    head_width=0.02,
                    head_length=0.02,
                    fc=self.plot.palette[0],
                    color=self.plot.palette[0],
                    transform=_ax.transAxes,
                )
            _ax.text(
                0.5,
                0,
                (x if self.plot_params.xlabel is None else self.plot_params.xlabel),
                transform=_ax.transAxes,
                va="top",
                ha="center",
                fontsize="x-small",
            )
            _ax.text(
                0,
                0.5,
                (y if self.plot_params.xlabel is None else self.plot_params.ylabel),
                transform=_ax.transAxes,
                va="center",
                ha="right",
                rotation="vertical",
                fontsize="x-small",
            )
            _ax.set_xticks([])
            _ax.set_yticks([])
            sns.despine(ax=_ax, left=True, right=True, top=True, bottom=True)

        if len(self.splits) > 1 and self.legend_loc:
            _patch_splits = [
                mpatches.Patch(color=_palette[_split], label=_split.title())
                for _split in self.splits
            ]
            _legend = g.figure.legend(
                handles=_patch_splits,
                loc=self.legend_loc,
                bbox_to_anchor=self.legend_bbox_to_anchor,
                ncols=(
                    len(_patch_splits)
                    if self.legend_ncols is None
                    else self.legend_ncols
                ),
                frameon=False,
            )
            _extra_artists.append(_legend)
            g.figure.set_size_inches(self.plot.get_size())
            _kwargs = {}
            if self.rasterize_scatter:
                _kwargs["dpi"] = 300
            if self.plot.file is not None:
                g.figure.savefig(
                    self.plot.file,
                    format=Path(self.plot.file.name).suffix[1:],
                    bbox_extra_artists=_extra_artists,
                    bbox_inches="tight",
                    **_kwargs,
                )
        else:
            g.figure.set_size_inches(self.plot.get_size())
            _kwargs = {}
            if self.rasterize_scatter:
                _kwargs["dpi"] = 300
            self._save_figure(g.figure, **_kwargs)
        return g.figure


class PlotClassExamples(BasePlotCmd):
    srcs: Tuple[Path]
    labels: Tuple[str]

    def __call__(self):
        assert len(self.srcs) == len(self.labels)
        _imgs = [mimg.imread(_src) for _src in self.srcs]
        with self.plot.open(
            nrows=1,
            ncols=len(_imgs),
            width_ratios=[_img.shape[1] for _img in _imgs],
            gridspec_kw={"wspace": 0.01},
        ) as (_fig, _axs):
            for _ax, _img, _label in zip(_axs, _imgs, self.labels):
                _ax.imshow(_img, interpolation="bicubic")
                _ax.annotate(
                    _label.title(), (0.5, -0.2), xycoords="axes fraction", ha="center"
                )

            for _ax in _axs:
                _ax.set_xticks([])
                _ax.set_yticks([])
                for spine in ["top", "right", "bottom", "left"]:
                    _ax.spines[spine].set_visible(True)
        return _fig


class PlotSplitAccVsRhosSingle(BasePlotCmd):
    res_dirs: Tuple[InputDirectory]
    res_files_pattern: str = "**/*.pth"
    eval_batch_size: int = 1024
    n_boot: int = 1000
    y: Literal["acc", "acc_delta"]
    legend_loc: str = "upper right"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None
    major_xticks: Optional[Tuple[float]] = None
    major_xticklabels: Optional[Tuple[str]] = None
    minor_xticks: Optional[Tuple[float]] = None

    def __call__(self):
        df_rows = []
        _dataset: Optional[SplitLTDataset] = None
        for _res_dir in self.res_dirs:
            for _res_file in tqdm(
                list(_res_dir.glob(self.res_files_pattern)),
                desc=f"Loading results from '{_res_dir}'",
                unit="file",
            ):
                _res = _load_train_res(_res_file, self.eval_batch_size)
                if _dataset is None:
                    _dataset = SplitLTDataset(_res.train_data_info.dataset_name)
                else:
                    assert str(_dataset) == _res.train_data_info.dataset_name
                _baseline_res = _load_baseline_res(_dataset, self.eval_batch_size)
                assert list(_res.training_config.sampler_builder.sampler_classes) == [
                    AllFewSampler,
                    ClassBalancedBaseSampler,
                ]
                _rho = float(_res.training_config.sampler_builder.sampler_args[1]["r"])
                for _split in ["Many", "Medium", "Few", "Overall"]:
                    df_rows.append(
                        {
                            "$\\rho$": _rho,
                            "Split": _split,
                            "Accuracy": _get_split_test_acc(_res, _split),
                            "Accuracy change": (
                                _get_split_test_acc(_res, _split)
                                - _get_split_test_acc(_baseline_res, _split)
                            ),
                        }
                    )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        with self.plot.open() as (_fig, _ax):
            sns.lineplot(
                df,
                x="$\\rho$",
                y=("Accuracy change" if self.y == "acc_delta" else "Accuracy"),
                hue="Split",
                style="Split",
                hue_order=["Overall", "Few", "Medium", "Many"],
                n_boot=self.n_boot,
                markers=True,
                ms=(mpl.rcParams["lines.markersize"] / 1.2),
                dashes=False,
                err_style="bars",
                ax=_ax,
                legend=True,
            )
            sns.despine(fig=_fig, ax=_ax, left=True, right=True, bottom=False, top=True)

            if self.y == "acc":
                _ax.set_ylim(0, 1.01)
                _ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            else:
                _ax.set_ylim(-0.25, 0.25)
                _ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
                _ax.axhline(y=0, ls="-", color=self.plot.palette[0], zorder=1)

            _ax.xaxis.set_tick_params(direction="in", which="both")
            if self.major_xticks is not None:
                _ax.xaxis.set_major_locator(FixedLocator(self.major_xticks))
                _ax.set_xlim(min(self.major_xticks), max(self.major_xticks))
            if self.major_xticklabels is not None:
                _ax.xaxis.set_major_formatter(FixedFormatter(self.major_xticklabels))
            if self.minor_xticks is not None:
                _ax.xaxis.set_minor_locator(FixedLocator(self.minor_xticks))

            if self.legend_loc:
                _legend = _ax.legend()
                _legend_handles = _legend.legendHandles
                _legend.remove()
                _ax.legend(
                    handles=_legend_handles,
                    ncols=4,
                    title="",
                    frameon=False,
                    loc=self.legend_loc,
                    bbox_to_anchor=self.legend_bbox_to_anchor,
                )

        return _fig


class PlotSplitAccVsExp(_BaseMultiExpPlotCmd, BasePlotCmd):
    col: Optional[Literal["dataset", "metric"]] = None
    col_wrap: Optional[int] = None
    y: Literal["acc", "acc_delta"]
    xlabel: str = ""
    legend_loc: str = "upper right"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None

    @staticmethod
    def _get_metric_desc(metric: Literal["euclidean", "cosine", "random"]) -> str:
        if metric == "random":
            return "Random neighbors"
        return f"{metric.title()} distance nearest neighbors"

    def __call__(self):
        df_rows = []
        for _fid, _exp_name, _dataset, _res in self._train_results():
            _baseline_res = _load_baseline_res(_dataset, self.eval_batch_size)
            for _split in ["Many", "Medium", "Few", "Overall"]:
                df_rows.append(
                    {
                        "FID": _fid,
                        "Experiment": _exp_name,
                        "Dataset": str(_dataset),
                        "Metric": self._get_metric_desc(_res.nn_info.nn_dist),
                        "Split": _split,
                        "Accuracy": _get_split_test_acc(_res, _split),
                        "Accuracy change": (
                            _get_split_test_acc(_res, _split)
                            - _get_split_test_acc(_baseline_res, _split)
                        ),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        if self.col is not None:
            self.col = self.col.title()
            _notcol = "Dataset" if self.col == "Metric" else "Metric"
            assert all(
                len(_s) == 1 for _s in df.groupby(self.col)[_notcol].agg(set).values
            )
            cols = list(df[self.col].unique())
            ncols = len(cols)
            if self.col_wrap is not None and ncols > self.col_wrap:
                nrows = ceil(ncols / self.col_wrap)
                ncols = self.col_wrap
            else:
                nrows = 1
        else:
            nrows = 1
            ncols = 1

        with self.plot.open(
            nrows=nrows, ncols=ncols, squeeze=False, gridspec_kw={"wspace": 0.05}
        ) as (_fig, _axs):
            for _i, (_col, _ax) in enumerate(
                itertools.zip_longest(cols, _axs.flatten())
            ):
                if _col is None and self.col is not None:
                    _ax.remove()
                    continue

                _col_df = df if self.col is None else df[df[self.col] == _col]
                sns.pointplot(
                    _col_df,
                    x="Experiment",
                    y=("Accuracy change" if self.y == "acc_delta" else "Accuracy"),
                    hue="Split",
                    hue_order=["Overall", "Few", "Medium", "Many"],
                    n_boot=self.n_boot,
                    units="FID",
                    dodge=0.4,
                    markers=["x", ".", "d", "*"],
                    ax=_ax,
                )
                sns.despine(
                    fig=_fig, ax=_ax, left=True, right=True, bottom=False, top=True
                )

                if self.col is not None and len(cols) > 1:
                    _ax.set_title(_col)

                _legend = _ax.legend()
                if _i == 0:
                    _legend_handles = _legend.legendHandles
                _legend.remove()

                if _i % ncols:
                    _ax.set_ylabel("")

                _ax.set_xlabel(self.xlabel)
                _ax.xaxis.set_tick_params(direction="in")

                if self.y == "acc":
                    _ax.set_ylim(0, 1.01)
                    _ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                else:
                    _ax.set_ylim(-0.25, 0.25)
                    _ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
                    _ax.axhline(y=0, ls="-", color=self.plot.palette[0], zorder=1)
                if _i != 0:
                    _ax.set_yticklabels([])

            if self.legend_loc:
                _fig.legend(
                    handles=_legend_handles,
                    ncols=4,
                    title="",
                    frameon=False,
                    loc=self.legend_loc,
                    bbox_to_anchor=self.legend_bbox_to_anchor,
                )
        return _fig


class PlotClsAccDeltaBySplit(_BaseMultiExpPlotCmd, BasePlotCmd):
    def __call__(self):
        df_rows = []
        test_acc_deltas__per__split__experiment = defaultdict(lambda: defaultdict(list))

        for _fid, _exp_name, _dataset, _res in self._train_results():
            _baseline_res = _load_baseline_res(_dataset, self.eval_batch_size)

            for _split in ("overall", "many", "medium", "few"):
                test_acc_deltas__per__split__experiment[_exp_name][_split].append(
                    _get_split_test_acc(_res, _split)
                    - _get_split_test_acc(_baseline_res, _split)
                )

            _delta_acc__per__class = {
                _class: (
                    _res.test_acc__per__class[_class]
                    - _baseline_res.test_acc__per__class[_class]
                )
                for _class in _res.test_acc__per__class
            }

            _class_idx = 0
            for _split in ("few", "medium", "many"):
                for _class in _res.train_data_info.class__set__per__split[_split]:
                    df_rows.append(
                        {
                            "FID": _fid,
                            "Experiment": _exp_name,
                            "Dataset": str(_dataset),
                            "Split": _split,
                            "Class index": _class_idx,
                            "Class": _class,
                            "Delta test accuracy": _delta_acc__per__class[_class],
                        }
                    )
                    _class_idx += 1

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:%s\n", df)

        assert all(
            len(_s) == 1 for _s in df.groupby("Experiment")["Dataset"].agg(set).values
        )

        _exps = list(test_acc_deltas__per__split__experiment.keys())
        _n_exps = len(_exps)

        with self.plot.open(
            close_fig_on_exit=False,
            nrows=_n_exps,
            ncols=1,
            squeeze=False,
            gridspec_kw={"hspace": 0.1},
        ) as (_fig, _axs):
            for _exp, _ax in zip(_exps, _axs[:, 0]):
                _exp_df = df[df["Experiment"] == _exp]
                _xticks = [0]
                _xorder = []
                for _split, _color in zip(
                    ("few", "medium", "many"), self.plot.palette[1:]
                ):
                    _split_df = _exp_df[_exp_df["Split"] == _split]
                    _xorder.extend(
                        _split_df[["Class", "Delta test accuracy"]]
                        .groupby("Class", group_keys=False, sort=False)
                        .mean()
                        .reset_index()
                        .sort_values("Delta test accuracy", ascending=False)["Class"]
                        .tolist()
                    )
                    _xticks.append(_split_df["Class index"].max())
                    _acc_del = mean(
                        test_acc_deltas__per__split__experiment[_exp][_split]
                    )
                    _ax.hlines(
                        y=_acc_del,
                        xmin=_xticks[-2],
                        xmax=_xticks[-1],
                        colors=[_color],
                        linestyles=["-"],
                        linewidths=[mpl.rcParams["lines.linewidth"] * 0.75],
                        zorder=2,
                        path_effects=[
                            pe.Stroke(
                                linewidth=(mpl.rcParams["lines.linewidth"] * 1.25),
                                foreground=self.plot.palette[0],
                                alpha=0.5,
                            ),
                            pe.Normal(),
                        ],
                    )
                    _ano = _ax.annotate(
                        f"$\\Delta={_acc_del:+.2f}$",
                        xy=(
                            _xticks[-1] * 0.98 if _acc_del > 0 else _xticks[-2] * 1.02,
                            _acc_del + 0.02 if _acc_del > 0 else _acc_del - 0.02,
                        ),
                        color=_color,
                        ha=("right" if _acc_del > 0 else "left"),
                        va=("bottom" if _acc_del > 0 else "top"),
                        fontsize="small",
                    )
                    _ano.set_path_effects(
                        [
                            pe.Stroke(
                                linewidth=(_ano.get_fontsize() / 15),
                                foreground=self.plot.palette[0],
                            ),
                            pe.Normal(),
                        ]
                    )
                    if _exp == _exps[-1]:
                        _ax.text(
                            _split_df["Class index"].mean(),
                            -0.7,
                            f"{_split.title()} split",
                            horizontalalignment="center",
                        )

                sns.barplot(
                    _exp_df,
                    x="Class",
                    y="Delta test accuracy",
                    hue="Split",
                    order=_xorder,
                    hue_order=("few", "medium", "many"),
                    errorbar=None,
                    units="FID",
                    palette=self.plot.palette[1:],
                    saturation=1,
                    width=1,
                    dodge=False,
                    ax=_ax,
                    linewidth=0,
                )

                _overall_del = mean(
                    test_acc_deltas__per__split__experiment[_exp]["overall"]
                )

                _title_text = _exp + "\n" if _n_exps > 1 else ""
                _title_text += (
                    f"Change in overall accuracy, $\\Delta={_overall_del:+.2f}$"
                )
                _ax.text(
                    0.95,
                    0.75,
                    _title_text,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    transform=_ax.transAxes,
                )

                _ax.set_xticks(_xticks[1:-1])
                _ax.set_xticklabels(["", ""])
                _ax.set_xlabel("")
                _ax.set_ylim(-0.6, 0.6)
                _ax.set_ylabel("Accuracy change")
                _ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
                _ax.axhline(y=0, ls="-", color=self.plot.palette[0], alpha=0.5)
                _ax.axhline(
                    y=_overall_del, ls="--", color=self.plot.palette[0], zorder=1
                )
                sns.despine(
                    ax=_ax,
                    left=(_n_exps == 1),
                    bottom=False,
                    right=(_n_exps == 1),
                    top=(_n_exps == 1),
                )
                _ax.legend().remove()
        return _fig


class PlotClsAccAndSamples(_BaseMultiExpPlotCmd, BasePlotCmd):
    col_wrap: Optional[int] = None

    def __call__(self):
        df_rows = []
        dataset_name__per__exp = {}
        class_order__per__exp = {}
        n_train__per__class__exp = defaultdict(lambda: defaultdict(int))

        for _gid, _exp_name, _dataset, _res in self._train_results():
            try:
                if dataset_name__per__exp[_exp_name] != str(_dataset):
                    raise AssertionError
            except KeyError:
                dataset_name__per__exp[_exp_name] = str(_dataset)
                _add_baseline_res = True
                _baseline_res = _load_baseline_res(_dataset, self.eval_batch_size)
            else:
                _add_baseline_res = False
                _baseline_res = None

            if _add_baseline_res:
                class_order__per__exp[_exp_name] = _get_classes_ordered_by_size(
                    _baseline_res, reverse=True
                )
            _train_imgs__per__cls = _res.train_data_info.n_imgs__per__class

            for _class in class_order__per__exp[_exp_name]:
                df_rows.append(
                    {
                        "GID": _gid,
                        "Experiment": _exp_name,
                        "Dataset": _dataset.proper_name,
                        "Class": _class,
                        "Model": "AlphaNet",
                        "Test accuracy": _res.test_acc__per__class[_class],
                        "Train samples": _train_imgs__per__cls[_class],
                    }
                )
                if _add_baseline_res:
                    n_train__per__class__exp[_exp_name][_class] = _train_imgs__per__cls[
                        _class
                    ]
                    df_rows.append(
                        {
                            "Experiment": _exp_name,
                            "Dataset": _dataset.proper_name,
                            "Class": _class,
                            "Model": "Baseline",
                            "Test accuracy": _baseline_res.test_acc__per__class[_class],
                            "Train samples": _train_imgs__per__cls[_class],
                        }
                    )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        _exps = list(dataset_name__per__exp.keys())
        _n_exps = len(_exps)

        _n_cols = _n_exps
        if self.col_wrap is not None and _n_cols > self.col_wrap:
            _n_rows = ceil(_n_cols / self.col_wrap)
            _n_cols = self.col_wrap
        else:
            _n_rows = 1

        with self.plot.open(
            close_fig_on_exit=False,
            nrows=_n_rows,
            ncols=_n_cols,
            squeeze=False,
            sharey=True,
            gridspec_kw={"hspace": 0.2},
        ) as (_fig, _axs):
            for _i, (_exp, _ax) in enumerate(
                itertools.zip_longest(_exps, _axs.flatten())
            ):
                if _exp is None:
                    _ax.remove()
                    continue

                _exp_df = df[df["Experiment"] == _exp]
                for _model in ["AlphaNet", "Baseline"]:
                    _model_df = _exp_df[_exp_df["Model"] == _model]
                    _color = self.plot.palette[int(_model == "AlphaNet")]
                    sns.barplot(
                        _model_df,
                        x="Class",
                        order=class_order__per__exp[_exp],
                        y="Test accuracy",
                        n_boot=self.n_boot,
                        units="GID",
                        saturation=1,
                        errorbar=None,
                        ax=_ax,
                        linewidth=None,
                        fill=True,
                        color=_color,
                        edgecolor=_color,
                        facecolor=_color,
                        alpha=(0.8 if _model == "AlphaNet" else 0.6),
                        label=_model,
                    )

                sns.despine(ax=_ax, left=False, top=True, right=True, bottom=True)
                _ax.grid(which="major", visible=False)
                _ax.tick_params(size=mpl.rcParams["xtick.major.size"])
                if _i == _n_cols - 1:
                    _ax.legend(loc="upper right", frameon=False)
                if _n_exps > 1:
                    _ax.set_title(_exp, y=0.85)

                _ax.set_xticks([])
                _ax.set_xlim(0, df["Class"].max())
                _ax.set_xlabel("")

                _ax.set_ylim(-0.05, 1.05)
                _ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                if _i % _n_cols:
                    _ax.set_ylabel("")
                    _ax.tick_params(labelleft=False)
                else:
                    _ax.set_ylabel("Test accuracy")

                _ax2 = _ax.twinx()
                _ax2.plot(
                    class_order__per__exp[_exp],
                    [
                        n_train__per__class__exp[_exp][_c]
                        for _c in class_order__per__exp[_exp]
                    ],
                    color=self.plot.palette[2],
                )

                sns.despine(ax=_ax2, left=True, right=False, top=True, bottom=True)
                _ax2.grid(which="major", visible=False)
                _ax2.tick_params(
                    color=self.plot.palette[2],
                    labelcolor=self.plot.palette[2],
                    size=mpl.rcParams["xtick.major.size"],
                    right=True,
                )

                _ax2.set_xticks([])
                _ax2.set_xlabel("")

                _ax2.yaxis.set_major_locator(MultipleLocator(100))
                if _i % _n_cols == _n_cols - 1 or _i == _n_exps - 1:
                    _ax2.set_ylabel("Train samples", color=self.plot.palette[2])
                else:
                    _ax2.set_ylabel("")
                    _ax2.tick_params(labelright=False)
        return _fig


class PlotAlphaDist(_BaseMultiExpPlotCmd, BasePlotCmd):
    col_wrap: Optional[int] = None
    legend_loc: Optional[str] = "center right"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None
    legend_ncols: Optional[int] = None
    source_idxs: Optional[Tuple[int, ...]] = None

    def __call__(self):
        df_rows = []
        for _gid, _exp_name, _dataset, _res in self._train_results():
            _alphanet_classifier = _res.load_best_alphanet_classifier()
            _alphanet_classifier = _alphanet_classifier.to(DEFAULT_DEVICE).eval()
            _alpha__mat = _alphanet_classifier.get_learned_alpha_vecs()
            assert all(bool(_a0 == 1) for _a0 in _alpha__mat[:, 0])

            _source_idxs = (
                range(1, _alpha__mat.shape[1])
                if self.source_idxs is None
                else self.source_idxs
            )
            for _target, _source in itertools.product(
                range(_alpha__mat.shape[0]), _source_idxs
            ):
                if _source >= _alpha__mat.shape[1]:
                    continue
                df_rows.append(
                    {
                        "GID": _gid,
                        "Experiment": _exp_name,
                        "Target": _target,
                        "Source": f"$\\alpha_{{{_source}}}$",
                        "Alpha": _alpha__mat[_target, _source].item(),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        self.plot.config()
        _exps = list(df["Experiment"].unique())
        _n_exps = len(_exps)
        _n_srcs = len(df["Source"].unique())
        _palette = sns.cubehelix_palette(
            n_colors=_n_srcs,
            hue=1,
            gamma=0.5,
            dark=0.05,
            light=0.75,
            reverse=self.plot.theme == "light",
        )
        g = sns.displot(
            df,
            x="Alpha",
            hue="Source",
            col="Experiment",
            col_wrap=self.col_wrap,
            kind="kde",
            legend=True,
            palette=_palette,
            aspect=1,
            facet_kws={"sharex": True, "sharey": False},
        )
        if _n_exps > 1:
            g.set_titles("{col_name}")
        else:
            g.set_titles("")
        g.set_xlabels("")
        g.set_ylabels("")
        g.set(xlim=[-1, 1], xticks=[-1, 0, 1], yticks=[])
        g.despine(top=True, left=True, right=True, bottom=False, trim=True)
        g.tick_params(
            axis="x",
            which="major",
            reset=True,
            top=False,
            bottom=True,
            grid_linewidth=0,
        )
        if self.legend_loc:
            if self.legend_ncols is not None:
                _kwargs = {"ncols": self.legend_ncols}
            else:
                _kwargs = {}
            sns.move_legend(
                g,
                loc=self.legend_loc,
                title="",
                bbox_to_anchor=self.legend_bbox_to_anchor,
                **_kwargs,
            )
        else:
            g.legend.remove()
        g.figure.set_size_inches(self.plot.get_size())
        self._save_figure(g.figure)
        return g.figure


class PlotClsAccVsSamples(_BaseMultiExpPlotCmd, BasePlotCmd):
    col_wrap: Optional[int] = None
    use_husl: bool = False
    ci: Optional[int] = None
    legend_loc: Optional[str] = None
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None

    def __call__(self):
        df_rows = []
        dataset__set = set()
        for _gid, _exp_name, _dataset, _res in self._train_results():
            dataset__set.add(_dataset)
            for _class, _n_class in _res.train_data_info.n_imgs__per__class.items():
                df_rows.append(
                    {
                        "GID": _gid,
                        "Experiment": _exp_name,
                        "Dataset": _dataset.proper_name,
                        "Class": _class,
                        "Test accuracy": _res.test_acc__per__class[_class],
                        "Train samples": _n_class,
                    }
                )

        baseline_df_rows = []
        for _dataset in dataset__set:
            _baseline_res = _load_baseline_res(_dataset, self.eval_batch_size)
            for (
                _class,
                _n_class,
            ) in _baseline_res.train_data_info.n_imgs__per__class.items():
                baseline_df_rows.append(
                    {
                        "Experiment": "Baseline",
                        "Dataset": _dataset.proper_name,
                        "Class": _class,
                        "Test accuracy": _baseline_res.test_acc__per__class[_class],
                        "Train samples": _n_class,
                    }
                )

        df = pd.DataFrame(baseline_df_rows + df_rows)
        logging.info("loaded dataframe:\n%s", df)

        self.plot.config()
        g = sns.lmplot(
            data=df,
            x="Train samples",
            y="Test accuracy",
            hue="Experiment",
            col="Dataset",
            col_wrap=self.col_wrap,
            units="Class",
            logx=True,
            scatter=False,
            legend=True,
            palette=("husl" if self.use_husl else self.plot.palette),
            ci=self.ci,
        )

        g.despine(left=True, right=True, top=True, bottom=False)
        g.set(ylim=[0, 1.01], yticks=[0.2, 0.4, 0.6, 0.8, 1.0])
        g.legend.set_title("")
        if self.legend_loc is not None:
            sns.move_legend(
                g, self.legend_loc, bbox_to_anchor=self.legend_bbox_to_anchor
            )
        if len(dataset__set) > 1:
            g.set_titles("{col_var}")
        else:
            g.set_titles("")
        for _ax in g.axes.flat:
            _ax.set_xscale("log")

        g.figure.set_size_inches(self.plot.get_size())
        self._save_figure(g.figure)
        return g.figure


class PlotTemplateDeltas(BasePlotCmd):
    res_file: InputBinFile
    eval_batch_size: int = 1024
    delta_type: Literal["min", "max"] = "max"
    n_deltas: int = 1
    n_pcs: int = 50
    mds_iters: int = 300
    plot: ContextPlot

    def __call__(self):
        # Load result and baseline templates.
        alphanet_res = TrainResult.from_dict(
            torch.load(self.res_file, map_location=DEFAULT_DEVICE)
        )
        if alphanet_res.training_config.n_neighbors > len(self.plot.palette) - 1:
            raise ValueError(
                f"cannot handle results with more than "
                f"'{len(self.plot.palette)}' nearest neighbors"
            )

        dataset = SplitLTDataset(alphanet_res.train_data_info.dataset_name)
        baseline_res = TrainResult.from_dict(
            torch.load(dataset.baseline_eval_file_path, map_location=DEFAULT_DEVICE)
        )

        _alphanet_classifier = alphanet_res.load_best_alphanet_classifier()
        _alphanet_classifier = _alphanet_classifier.to(DEFAULT_DEVICE).eval()
        alphanet_template__mat = _alphanet_classifier.get_trained_templates()

        _baseline_clf = dataset.load_classifier(DEFAULT_DEVICE)
        baseline_template__mat = _baseline_clf.weight.data.detach()

        assert (
            baseline_template__mat.shape
            == alphanet_template__mat.shape
            == (
                alphanet_res.train_data_info.n_classes,
                alphanet_res.train_data_info.n_features,
            )
        )

        # Get test set predictions.
        (
            alphanet_test_acc__per__class,
            alphanet_test_pred__seq,
        ) = _get_test_acc_per_class(
            alphanet_res, self.eval_batch_size, return_preds=True
        )
        (
            baseline_test_acc__per__class,
            baseline_test_pred__seq,
        ) = _get_test_acc_per_class(
            baseline_res, self.eval_batch_size, return_preds=True
        )
        delta_test_acc__per__class = {
            _c: alphanet_test_acc__per__class[_c] - baseline_test_acc__per__class[_c]
            for _c in alphanet_test_acc__per__class
        }

        # Create ordered vectors of 'base' and 'few' split classes.
        bclass_ordered__vec = torch.tensor(
            list(
                alphanet_res.train_data_info.class__set__per__split["many"]
                | alphanet_res.train_data_info.class__set__per__split["medium"]
            ),
            device=DEFAULT_DEVICE,
        )
        fclass_ordered__vec = torch.tensor(
            list(alphanet_res.train_data_info.class__set__per__split["few"]),
            device=DEFAULT_DEVICE,
        )

        # Separate 'base' and 'few' split templates.
        btemplate__mat = torch.index_select(
            baseline_template__mat, 0, bclass_ordered__vec
        )
        assert torch.allclose(
            btemplate__mat,
            torch.index_select(alphanet_template__mat, 0, bclass_ordered__vec),
        )
        baseline_ftemplate__mat = torch.index_select(
            baseline_template__mat, 0, fclass_ordered__vec
        )
        alphanet_ftemplate__mat = torch.index_select(
            alphanet_template__mat, 0, fclass_ordered__vec
        )

        # Load projections of test samples.
        test_data = dataset.load_data(alphanet_res.training_config.test_datagrp)
        test_proj__mat = test_data.feat__mat

        # Normalize the templates and projections.
        # w.x + b =equiv= (w/p).(x/q) + (b/pq)
        _mean_template_norm = baseline_template__mat.norm(p=2, dim=1).mean()  # p
        _mean_test_proj_norm = test_proj__mat.norm(p=2, dim=1).mean()  # q
        norm_baseline_template__mat = baseline_template__mat / _mean_template_norm
        btemplate__mat /= _mean_template_norm
        baseline_ftemplate__mat /= _mean_template_norm
        alphanet_ftemplate__mat /= _mean_template_norm
        test_proj__mat /= _mean_test_proj_norm

        # Perform PCA for initial dimensionality reduction.
        _pca_pipe = make_pipeline(StandardScaler(), PCA(self.n_pcs))
        _pca_pipe.fit(norm_baseline_template__mat.numpy(force=True))
        btemplate_pcs__nmat = _pca_pipe.transform(btemplate__mat.numpy(force=True))
        baseline_ftemplate_pcs__nmat = _pca_pipe.transform(
            baseline_ftemplate__mat.numpy(force=True)
        )
        alphanet_ftemplate_pcs__nmat = _pca_pipe.transform(
            alphanet_ftemplate__mat.numpy(force=True)
        )
        test_proj_pcs__nmat = _pca_pipe.transform(test_proj__mat.numpy(force=True))

        # Use MDS to reduce templates and projections to 2D.
        _alldata__nmat = np.vstack(
            (
                btemplate_pcs__nmat,
                baseline_ftemplate_pcs__nmat,
                alphanet_ftemplate_pcs__nmat,
                test_proj_pcs__nmat,
            )
        )
        _mds = MDS(
            2, verbose=2, n_jobs=-1, normalized_stress="auto", max_iter=self.mds_iters
        )
        _alldata_embed__nmat = _mds.fit_transform(_alldata__nmat)
        _alldata_embed__mat__seq = [None, None, None, None]
        _n = 0
        for _i, _mat in enumerate(
            [
                btemplate__mat,
                baseline_ftemplate__mat,
                alphanet_ftemplate__mat,
                test_proj__mat,
            ]
        ):
            _embed__mat = torch.from_numpy(
                _alldata_embed__nmat[_n : _n + _mat.shape[0]]
            ).to(device=DEFAULT_DEVICE)
            _alldata_embed__mat__seq[_i] = _embed__mat
            _n += _mat.shape[0]
        (
            btemplate_embed__mat,
            baseline_ftemplate_embed__mat,
            alphanet_ftemplate_embed__mat,
            test_proj_embed__mat,
        ) = _alldata_embed__mat__seq

        # Get 'n_deltas' 'few' split classes based on 'delta_type' change in accuracy.
        _fun = nsmallest if self.delta_type == "min" else nlargest
        selected_fclass__seq = _fun(
            self.n_deltas,
            alphanet_res.train_data_info.class__set__per__split["few"],
            key=lambda _c: delta_test_acc__per__class[_c],
        )
        selected_fclass__seq = list(
            sorted(
                selected_fclass__seq,
                key=lambda _c: delta_test_acc__per__class[_c],
                reverse=(self.delta_type == "max"),
            )
        )

        # Get indexes for the selected classes.
        _idx__per__fclass = {
            _fclass_tensor.item(): _i
            for _i, _fclass_tensor in enumerate(fclass_ordered__vec)
        }
        selected_idx__seq = [
            _idx__per__fclass[_fclass] for _fclass in selected_fclass__seq
        ]

        # Get template embeddings for the selected classes.
        (
            selected_baseline_ftemplate_embed__mat,
            selected_alphanet_ftemplate_embed__mat,
        ) = (
            torch.index_select(
                _ftemplate_embed__mat,
                0,
                torch.tensor(selected_idx__seq, device=DEFAULT_DEVICE),
            )
            for _ftemplate_embed__mat in [
                baseline_ftemplate_embed__mat,
                alphanet_ftemplate_embed__mat,
            ]
        )

        # Get nearest neighbor template embeddings for the selected classes.
        _idx__per__bclass = {
            _bclass_tensor.item(): _i
            for _i, _bclass_tensor in enumerate(bclass_ordered__vec)
        }
        nn_btemplate_embed__mat__per__selected_fclass = {}
        for _fclass in selected_fclass__seq:
            _nn_class__seq = alphanet_res.nn_info.nn_class__seq__per__fclass[_fclass]
            _nn_idx__seq = [_idx__per__bclass[_bclass] for _bclass in _nn_class__seq]
            nn_btemplate_embed__mat__per__selected_fclass[_fclass] = torch.index_select(
                btemplate_embed__mat,
                0,
                torch.tensor(_nn_idx__seq, device=DEFAULT_DEVICE),
            )

        # Get test projection embeddings for the selected classes.
        _test_idx__seq__per__selected_fclass = defaultdict(list)
        for _i, _label in enumerate(test_data.label__seq):
            if _label in selected_fclass__seq:
                _test_idx__seq__per__selected_fclass[_label].append(_i)

        test_proj_embed__mat__per__selected_fclass = {}
        alphanet_test_pred__seq__per__selected_fclass = defaultdict(list)
        baseline_test_pred__seq__per__selected_fclass = defaultdict(list)
        for _fclass, _test_idx__seq in _test_idx__seq__per__selected_fclass.items():
            test_proj_embed__mat__per__selected_fclass[_fclass] = torch.index_select(
                test_proj_embed__mat,
                0,
                torch.tensor(_test_idx__seq, device=DEFAULT_DEVICE),
            )
            (
                alphanet_test_pred__seq__per__selected_fclass[_fclass],
                baseline_test_pred__seq__per__selected_fclass[_fclass],
            ) = [
                # pylint: disable=unsubscriptable-object
                [_test_pred__seq[_i] for _i in _test_idx__seq]
                for _test_pred__seq in [
                    alphanet_test_pred__seq,
                    baseline_test_pred__seq,
                ]
            ]

        # Plot.
        with self.plot.open(
            close_fig_on_exit=False,
            ncols=2,
            nrows=self.n_deltas,
            sharex="row",
            sharey="row",
            squeeze=False,
        ) as (_fig, _ax__mat):
            _ftemplate_dot_rad = mpl.rcParams["lines.markersize"] * 3
            _test_proj_dot_rad = _ftemplate_dot_rad / 5
            _n_neighbors = alphanet_res.training_config.n_neighbors
            _nn_dot_rad__seq = torch.linspace(
                _test_proj_dot_rad, _ftemplate_dot_rad, _n_neighbors + 2
            )[1:-1]

            for _n in range(self.n_deltas):
                # Plot the 'few' split class with the '_n'th <largest/smallest>
                # change in accuracy.
                _ax__vec = _ax__mat[_n, :]
                _fclass = selected_fclass__seq[_n]
                _test_proj__mat = test_proj_embed__mat__per__selected_fclass[_fclass]
                assert _test_proj__mat.shape == (
                    test_data.info.n_imgs__per__class[_fclass],
                    2,
                )
                _nn_embed__mat = nn_btemplate_embed__mat__per__selected_fclass[_fclass]
                assert _nn_embed__mat.shape == (_n_neighbors, 2)

                for _ax, _ftemplate_embed__vec, _test_pred__seq, _res_type in zip(
                    _ax__vec,
                    [
                        selected_baseline_ftemplate_embed__mat[_n, :],
                        selected_alphanet_ftemplate_embed__mat[_n, :],
                    ],
                    [
                        baseline_test_pred__seq__per__selected_fclass[_fclass],
                        alphanet_test_pred__seq__per__selected_fclass[_fclass],
                    ],
                    ["baseline", "alphanet"],
                ):
                    # Plot the 'few' split clsss template embedding.
                    assert _ftemplate_embed__vec.shape == (2,)
                    _color = "#0008" if _res_type == "baseline" else "#fffb"
                    _ax.scatter(
                        [_ftemplate_embed__vec[0].item()],
                        [_ftemplate_embed__vec[1].item()],
                        s=(_ftemplate_dot_rad**2),
                        c=_color,
                        edgecolors=self.plot.palette[0],
                        marker="*",
                        zorder=2,
                    )
                    if _res_type == "alphanet":
                        # Plot the baseline template also.
                        _embed__vec = selected_baseline_ftemplate_embed__mat[_n, :]
                        _ax.scatter(
                            [_embed__vec[0].item()],
                            [_embed__vec[1].item()],
                            s=(_ftemplate_dot_rad**2),
                            c="#0008",
                            edgecolors=self.plot.palette[0],
                            marker="*",
                            zorder=1,
                        )

                    # Plot the test projection embeddings.
                    for _test_proj__vec, _test_pred in zip(
                        _test_proj__mat, _test_pred__seq
                    ):
                        assert _test_proj__vec.shape == (2,)
                        if _test_pred == _fclass:
                            _color = self.plot.palette[0]
                        else:
                            try:
                                _idx = (
                                    alphanet_res.nn_info.nn_class__seq__per__fclass[
                                        _fclass
                                    ]
                                ).index(_test_pred)
                                _color = self.plot.palette[1 + _idx]
                            except ValueError:
                                _color = "#aaa"

                        _ax.scatter(
                            [_test_proj__vec[0].item()],
                            [_test_proj__vec[1].item()],
                            s=(_test_proj_dot_rad**2),
                            c=_color,
                            marker=".",
                            zorder=-1,
                        )

                    # Plot the nearest neighbor template embeddings.
                    for _j, _nn_embed__vec in enumerate(_nn_embed__mat):
                        assert _nn_embed__vec.shape == (2,)
                        _edgecolors = self.plot.palette[1 + _j]
                        _color = _edgecolors + "bb"
                        _dot_rad = _nn_dot_rad__seq[_j]
                        _ax.scatter(
                            [_nn_embed__vec[0].item()],
                            [_nn_embed__vec[1].item()],
                            s=(_dot_rad**2),
                            c=_color,
                            edgecolors=self.plot.palette[0],
                            marker="X",
                            zorder=0,
                        )

                    _ax.set_xticks([])
                    _ax.set_yticks([])
                    _ax.grid(visible=False)
                    sns.despine(
                        left=False, right=False, top=False, bottom=False, ax=_ax
                    )
        return _fig


class PlotPredCounts(_BaseMultiExpPlotCmd, BasePlotCmd):
    col_wrap: Optional[int] = None
    split: Literal["few", "base"] = "few"
    nn_split: Literal["few", "base", "all"] = "base"

    @staticmethod
    def _get_pred_status(_y, _yhat, _nns__per__class):
        if _yhat == _y:
            return "Correct"
        if _yhat in _nns__per__class[_y]:
            return "Incorrect as NN"
        return "Incorrect as non-NN"

    @staticmethod
    def _get_status_label(status):
        if status == "Correct":
            return "Correctly\nclassified"
        if status == "Incorrect as NN":
            return "Incorrectly\nclassified\nas a NN"
        return "Incorrectly\nclassified as\na non-NN"

    def __call__(self):
        df_rows = []

        for _gid, _exp_name, _dataset, _res in self._train_results():
            _, _res_test_yhats = _get_test_acc_per_class(
                _res, self.eval_batch_size, return_preds=True
            )
            _test_datagrp = _TEST_DATA_CACHE[str(_dataset)]
            _test_ys = _test_datagrp.label__seq

            _, _nn__seq__per__split_class = _get_nn_dist_per_split_class(
                _dataset,
                _res.nn_info.nn_dist,
                _res.nn_info.n_neighbors,
                for_split=self.split,
                against_split=self.nn_split,
            )

            for _i, (_y, _res_yhat) in enumerate(zip(_test_ys, _res_test_yhats)):
                if _y not in _nn__seq__per__split_class:
                    continue
                df_rows.append(
                    {
                        "GID": _gid,
                        "Dataset": _dataset.proper_name,
                        "Experiment": _exp_name,
                        "Sample": _i,
                        "Prediction status": self._get_status_label(
                            self._get_pred_status(
                                _y, _res_yhat, _nn__seq__per__split_class
                            )
                        ),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)
        assert all(
            len(_s) == 1
            for _s in df.groupby("Experiment")["Dataset"].aggregate(set).values
        )

        self.plot.config()
        g = sns.catplot(
            df,
            x="Prediction status",
            order=list(
                map(
                    self._get_status_label,
                    ["Correct", "Incorrect as NN", "Incorrect as non-NN"],
                )
            ),
            col="Dataset",
            col_wrap=self.col_wrap,
            estimator=None,
            errorbar=None,
            kind="count",
            palette=[self.plot.palette[3], self.plot.palette[2], self.plot.palette[1]],
            facet_kws={"sharex": True, "sharey": True},
        )
        if len(df["Dataset"].unique()) > 1:
            g.set_titles("{col_name}")
        else:
            g.set_titles("")
        for facet_name, ax in g.axes_dict.items():
            ax_df = df[df["Dataset"] == facet_name]
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(ax_df)))
            ax.set_yticks([int(len(ax_df) * _p / 100) for _p in [15, 30, 45]])
        g.set_xlabels("")
        g.set_ylabels("")
        g.despine(top=True, left=True, right=True, bottom=False, trim=False)
        g.tick_params(axis="x", which="major", top=False, bottom=False)
        g.figure.set_size_inches(self.plot.get_size())
        self._save_figure(g.figure)
        return g.figure


class PlotPredChanges(_BaseMultiExpPlotCmd, BasePlotCmd):
    col_wrap: Optional[int] = None
    label_rot: float = 0
    label_size: Literal[
        "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"
    ] = "x-small"
    split: Literal["few", "base", "all"] = "few"
    nn_split: Literal["few", "base", "all", "semantic"] = "base"
    semantic_nns_level: Optional[int] = None
    imagenet_data_root: Optional[Path] = None
    save_semantic_nns_file: Optional[OutputTextFile] = None

    @staticmethod
    def _get_pred_status(_y, _yhat, _nns__per__class):
        if _yhat == _y:
            return "Correct"
        if _yhat in _nns__per__class[_y]:
            return "Incorrect as NN"
        return "Incorrect as non-NN"

    @staticmethod
    def _get_status_label(status):
        if status == "Correct":
            return "Correctly\nclassified"
        if status == "Incorrect as NN":
            return "Incorrectly\nclassified\nas a NN"
        return "Incorrectly\nclassified as\na non-NN"

    @staticmethod
    def _make_band(xl, xr, yl, yr, w):
        assert xr > xl
        assert w > 0

        if torch.isclose(torch.tensor(data=float(yr)), torch.tensor(data=float(yl))):
            return mpath.Path([(xl, yl), (xr, yr), (xr, yr + w), (xl, yl + w)])

        y_del = abs(yr - yl)
        x_del = xr - xl

        x_del2 = x_del * x_del
        y_del2 = y_del * y_del

        q = w * abs(x_del2 - y_del2) / (x_del2 + y_del2)
        p = (w + y_del - q) / 2
        r = w * (y_del - p) / ((2 * p) - y_del)
        assert r > 0

        a = r * x_del / (w + (2 * r))
        b = x_del - (2 * a)
        assert b > 0

        t = math.asin(a / r)
        assert 0 < t < (math.pi / 2)
        z = math.tan(t / 2)
        bz_del1 = r * z
        bz_del2 = bz_del1 + (w * z)

        if yl > yr:
            yl, yr = yl + w, yr + w
            w, p, q = -w, -p, -q

        return mpath.Path(
            [
                (xl, yl),
                (xl + bz_del2, yl),
                (xl + a + b, yl + p),
                (xr - bz_del1, yr),
                (xr, yr),
                (xr, yr + w),
                (xr - bz_del2, yr + w),
                (xl + a, yl + p + q),
                (xl + bz_del1, yl + w),
                (xl, yl + w),
                (xl, yl),
            ],
            [
                mpath.Path.MOVETO,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.LINETO,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.LINETO,
            ],
        )

    def __call__(self):
        df_rows = []
        baseline_test_yhats__per__dataset = {}

        if self.nn_split == "semantic":
            assert self.semantic_nns_level is not None
            assert self.imagenet_data_root is not None
            wordnet_nn__seq__per__split_class = get_wordnet_nns_per_imgnet_class(
                self.semantic_nns_level,
                self.split,
                str(self.imagenet_data_root / "splits" / "few.txt"),
                str(self.imagenet_data_root / "label_names.txt"),
                str(self.imagenet_data_root / "labels_full.txt"),
                self.save_semantic_nns_file,
            )

        for _gid, _exp_name, _dataset, _res in self._train_results():
            if self.nn_split == "semantic":
                assert str(_dataset).startswith("imagenetlt")
            try:
                _baseline_test_yhats = baseline_test_yhats__per__dataset[_dataset]
            except KeyError:
                _baseline_res = TrainResult.from_dict(
                    torch.load(
                        _dataset.baseline_eval_file_path, map_location=DEFAULT_DEVICE
                    )
                )
                _, _baseline_test_yhats = _get_test_acc_per_class(
                    _baseline_res, self.eval_batch_size, return_preds=True
                )
                baseline_test_yhats__per__dataset[_dataset] = _baseline_test_yhats

            _, _res_test_yhats = _get_test_acc_per_class(
                _res, self.eval_batch_size, return_preds=True
            )
            _test_datagrp = _TEST_DATA_CACHE[str(_dataset)]
            _test_ys = _test_datagrp.label__seq

            if self.nn_split == "semantic":
                # pylint: disable=used-before-assignment
                _nn__seq__per__split_class = wordnet_nn__seq__per__split_class
            else:
                _, _nn__seq__per__split_class = _get_nn_dist_per_split_class(
                    _dataset,
                    _res.nn_info.nn_dist,
                    _res.nn_info.n_neighbors,
                    for_split=self.split,
                    against_split=self.nn_split,
                )
            fclass__set = _res.train_data_info.class__set__per__split["few"]

            for _i, (_y, _baseline_yhat, _res_yhat) in enumerate(
                zip(_test_ys, _baseline_test_yhats, _res_test_yhats)
            ):
                if _y not in _nn__seq__per__split_class:
                    continue
                df_rows.append(
                    {
                        "GID": _gid,
                        "Dataset": _dataset.proper_name,
                        "Experiment": _exp_name,
                        "Sample": _i,
                        "Class": _y,
                        "mSplit": ("Few" if _y in fclass__set else "Base"),
                        "Baseline prediction": self._get_pred_status(
                            _y, _baseline_yhat, _nn__seq__per__split_class
                        ),
                        "AlphaNet prediction": self._get_pred_status(
                            _y, _res_yhat, _nn__seq__per__split_class
                        ),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)
        assert all(
            len(_s) == 1
            for _s in df.groupby("Experiment")["Dataset"].aggregate(set).values
        )

        _exps = list(df["Experiment"].unique())
        _statuses = ["Incorrect as non-NN", "Incorrect as NN", "Correct"]
        if self.col_wrap is None or self.col_wrap >= len(_exps):
            _n_cols = len(_exps)
            _n_rows = 1
        else:
            _n_cols = self.col_wrap
            _n_rows = ceil(len(_exps) / self.col_wrap)

        if self.nn_split == "semantic":
            if self.plot.theme == "light":
                palette = ["#e5ae39", "#179be8", "#4f9e89"]
                hatch_palette = ["#ffd67f", "#7fc1e8", "#66ccb0"]
            else:
                palette = ["#15419e", "#ba5012", "#8d4e5e"]
                hatch_palette = ["#002166", "#663212", "#7a293f"]
        else:
            palette = self.plot.palette[1:4]
            if self.plot.theme == "light":
                hatch_palette = ["#e5cfa0", "#b9d7e8", "#00cc95"]
            else:
                hatch_palette = ["#101d39", "#2a180e", "#991f40"]

        with self.plot.open(
            close_fig_on_exit=False, nrows=_n_rows, ncols=_n_cols, squeeze=False
        ) as (_fig, _axs):
            for _axno, (_exp, _ax) in enumerate(
                itertools.zip_longest(_exps, _axs.flatten())
            ):
                if _exp is None:
                    _ax.remove()
                    continue

                _exp_df = df[df["Experiment"] == _exp]
                _n_preds = len(_exp_df)

                if len(_exps) > 1:
                    _ax.set_title(
                        _exp,
                        y=0.98,
                        verticalalignment="top",
                        pad=0,
                        color=self.plot.palette[0],
                    )
                sns.despine(ax=_ax, top=True, bottom=True, left=True, right=True)

                _status_to_baseline_n = {}
                _status_to_alphanet_n = {}
                _xcoords = [0.05, 0.45]
                _bar_width = 0.1

                for _i, _model in zip(_xcoords, ["Baseline", "AlphaNet"]):
                    _offset = 0
                    for _j, _status in enumerate(_statuses):
                        _n_status = len(
                            _exp_df[_exp_df[f"{_model} prediction"] == _status]
                        )
                        _bar_height = _n_status / _n_preds
                        _ax.bar(
                            _i,
                            _bar_height,
                            width=_bar_width,
                            bottom=_offset,
                            color=palette[_j],
                            linewidth=0,
                        )

                        if self.split == "all":
                            _n_few_status = len(
                                _exp_df[
                                    (_exp_df[f"{_model} prediction"] == _status)
                                    & (_exp_df["mSplit"] == "Few")
                                ]
                            )

                            _few_bar_height = _n_few_status / _n_preds
                            _ax.bar(
                                _i,
                                _few_bar_height,
                                width=_bar_width,
                                bottom=_offset,
                                color=palette[_j],
                                hatch="////",
                                linewidth=0,
                                edgecolor=hatch_palette[_j],
                            )

                        _y = _offset + (_bar_height / 2)
                        if (_axno % _n_cols) == 0 and _model == "Baseline":
                            _va = "top"
                        else:
                            _va = "center"
                        _ax.text(
                            _i,
                            _y,
                            str(_n_status),
                            ha="center",
                            va=_va,
                            fontsize=self.label_size,
                            color=self.plot.palette[0],
                            rotation=self.label_rot,
                        )

                        if (_axno % _n_cols) == 0 and _model == "Baseline":
                            _ax.text(
                                (_i - (_bar_width / 2) + (_bar_width / 7)),
                                _y + 0.01,
                                self._get_status_label(_status),
                                ha="left",
                                va="bottom",
                                fontsize=self.label_size,
                                color=self.plot.palette[0],
                                rotation=self.label_rot,
                            )

                        _offset += _bar_height
                        if _model == "Baseline":
                            _status_to_baseline_n[_status] = _n_status
                        else:
                            _status_to_alphanet_n[_status] = _n_status
                    assert torch.isclose(
                        torch.tensor(data=1.0), torch.tensor(data=_offset)
                    )

                _band_yl = 0
                _band_xl = _xcoords[0] + (_bar_width / 2)
                _band_xr = _xcoords[1] - (_bar_width / 2)
                _band_roffset__per__status = defaultdict(float)

                for _j, _lstatus in enumerate(_statuses):
                    _band_yrbase = 0.0
                    for _k, _rstatus in enumerate(_statuses):
                        _band_yr = _band_yrbase + _band_roffset__per__status[_rstatus]
                        _band_yrbase += _status_to_alphanet_n[_rstatus] / _n_preds
                        _n_l2r = len(
                            _exp_df[
                                (_exp_df["Baseline prediction"] == _lstatus)
                                & (_exp_df["AlphaNet prediction"] == _rstatus)
                            ]
                        )
                        _band_w = _n_l2r / _n_preds

                        if _n_l2r == 0:
                            continue

                        _band = self._make_band(
                            _band_xl, _band_xr, _band_yl, _band_yr, _band_w
                        )
                        _ax.add_patch(mpatches.PathPatch(_band, linewidth=0, fc="none"))
                        _grad_img = _gradient_image(
                            _ax,
                            extent=(
                                _band_xl,
                                _band_xr,
                                min(_band_yl, _band_yr),
                                max(_band_yl, _band_yr) + _band_w,
                            ),
                            direction=1,
                            cmap=sns.blend_palette(
                                [palette[_j], palette[_k]], as_cmap=True
                            ),
                            alpha=0.75,
                            aspect="auto",
                        )
                        _grad_img.set_clip_path(_band, transform=_ax.transData)

                        _band_yl += _band_w
                        _band_roffset__per__status[_rstatus] += _band_w

                    assert torch.isclose(
                        torch.tensor(data=float(_band_yl)),
                        torch.tensor(
                            data=float(
                                sum(
                                    _status_to_baseline_n[_s]
                                    for _s in _statuses[: _j + 1]
                                )
                                / _n_preds
                            )
                        ),
                    )
                    assert torch.isclose(
                        torch.tensor(data=_band_yrbase), torch.tensor(data=1.0)
                    )
                assert all(
                    torch.isclose(
                        torch.tensor(data=_band_roffset__per__status[_s]),
                        torch.tensor(data=_status_to_alphanet_n[_s] / _n_preds),
                    )
                    for _s in _statuses
                )
                assert torch.isclose(
                    torch.tensor(data=_band_yl), torch.tensor(data=1.0)
                )
                assert torch.isclose(
                    torch.tensor(data=_band_yr + _band_w), torch.tensor(data=1.0)
                )
                _ax.set_aspect(1)
                _ax.set_xlim(
                    _xcoords[0] - (_bar_width / 2), _xcoords[1] + (_bar_width / 2)
                )
                _ax.set_ylim(-0.01, 1.01)
                _ax.set_xticks([])
                _ax.set_yticks([])
                _ax.xaxis.grid(visible=False)
                _ax.yaxis.grid(visible=False)

                _rect = mpatches.Rectangle(
                    (_xcoords[0] - (_bar_width / 2) + 0.002, 0),
                    _xcoords[1] - _xcoords[0] + _bar_width - 0.004,
                    1,
                    fill=False,
                    lw=(1 * mpl.rcParams["lines.linewidth"]),
                    color=self.plot.palette[0],
                    transform=_ax.transData,
                )
                _ax.add_patch(_rect)
        return _fig
