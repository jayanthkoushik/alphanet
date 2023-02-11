import itertools
import logging
from collections import defaultdict
from heapq import nlargest, nsmallest
from math import ceil, sqrt
from pathlib import Path
from statistics import mean
from typing import Dict, Iterator, Literal, Optional, Tuple

import ignite.contrib.handlers
import ignite.engine
import ignite.metrics
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from corgy import Corgy
from corgy.types import InputBinFile, InputDirectory
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from alphanet._dataset import SplitLTDataGroup, SplitLTDataset
from alphanet._samplers import AllFewSampler, ClassBalancedBaseSampler
from alphanet._utils import ContextPlot, PlotParams
from alphanet.train import TrainResult

logging.root.setLevel(logging.INFO)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_train_res_with_per_cls_accs(res_file: Path, batch_size: int) -> TrainResult:
    res = TrainResult.from_dict(torch.load(res_file, map_location=DEFAULT_DEVICE))
    if res.test_acc__per__class is None:
        res.test_acc__per__class, _ = get_per_class_test_accs(res, batch_size)
        torch.save(res.as_dict(recursive=True), res_file)
    return res


def get_test_split_acc(res: TrainResult, split: str) -> float:
    if split.lower() == "overall":
        return res.test_metrics["accuracy"]
    return res.test_acc__per__split[split.lower()]


def get_metric_desc(metric: Literal["euclidean", "cosine", "random"]):
    if metric == "random":
        return "Random neighbors"
    return f"{metric.title()} distance nearest neighbors"


class BasePlotCmd(Corgy, corgy_make_slots=False):
    plot: ContextPlot

    def __call__(self):
        raise NotImplementedError

    def _get_ha(self) -> Dict[str, float]:
        _plot_size = self.plot.get_size()
        return {"height": _plot_size[1], "aspect": _plot_size[0] / _plot_size[1]}

    def _set_facet_grid_size(self, g: sns.FacetGrid) -> None:
        _n_rows, _n_cols = g.axes.shape
        if g._col_wrap is not None and g._col_wrap < _n_cols:
            _n_cols = g._col_wrap
            _n_rows = _n_rows * ceil(_n_cols / g._col_wrap)
        _plot_size = self.plot.get_size()
        g.figure.set_size_inches(_plot_size[0] * _n_cols, _plot_size[1] * _n_rows)

    def _save_facet_grid(self, g: sns.FacetGrid) -> None:
        if self.plot.file is not None:
            g.figure.savefig(
                self.plot.file, format=Path(self.plot.file.name).suffix[1:]
            )
            self.plot.file.close()


class _BaseMultiFilePlotCmd(Corgy, corgy_make_slots=False):
    base_res_dir: InputDirectory
    res_sub_dirs: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    eval_batch_size: int = 1024

    def _iter_train_results(self) -> Iterator[TrainResult]:
        dataset_names = set()
        for _res_sub_dir in self.res_sub_dirs or [""]:
            for _res_file in tqdm(
                list((self.base_res_dir / _res_sub_dir).glob(self.res_files_pattern)),
                desc=f"Loading {_res_sub_dir}".rstrip(),
                unit="file",
            ):
                res = load_train_res_with_per_cls_accs(_res_file, self.eval_batch_size)
                dataset_names.add(res.train_data_info.dataset_name)
                yield res


class _BaseMultiExpPlotCmd(Corgy, corgy_make_slots=False):
    base_res_dir: InputDirectory
    exp_sub_dirs: Tuple[str]
    exp_names: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    eval_batch_size: int = 1024
    col_wrap: Optional[int] = None
    n_boot: int = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._baseline_res_cache = {}

    def _iter_train_results(self) -> Iterator[Tuple[int, int, str, str, TrainResult]]:
        _exp_names = self.exp_sub_dirs if self.exp_names is None else self.exp_names
        if len(_exp_names) != len(self.exp_sub_dirs):
            raise ValueError("`exp_names` size mismatch with `exp_sub_dirs`")

        global_id = 0
        for _exp_sub_dir, exp_name in zip(self.exp_sub_dirs, _exp_names):
            for local_id, _res_file in enumerate(
                tqdm(
                    list(
                        (self.base_res_dir / _exp_sub_dir).glob(self.res_files_pattern)
                    ),
                    desc=f"Loading from '{_exp_sub_dir}'",
                    unit="file",
                )
            ):
                res = load_train_res_with_per_cls_accs(_res_file, self.eval_batch_size)
                dataset_name = res.train_data_info.dataset_name
                yield (global_id, local_id, exp_name, dataset_name, res)
                global_id += 1

    def _get_baseline_res(self, dataset_name: str) -> TrainResult:
        try:
            return self._baseline_res_cache[dataset_name]
        except KeyError:
            _baseline_res = load_train_res_with_per_cls_accs(
                SplitLTDataset(dataset_name).baseline_eval_file_path,
                self.eval_batch_size,
            )
            self._baseline_res_cache[dataset_name] = _baseline_res
            return _baseline_res


class PlotPerClsAccDeltaVsNNDist(_BaseMultiExpPlotCmd, BasePlotCmd):
    dist_for_rand: Literal["euclidean", "cosine"] = "euclidean"

    def __call__(self):
        df_rows = []
        for _gid, _lid, _exp_name, _dataset_name, _res in self._iter_train_results():
            _metric = _res.nn_info.nn_dist
            assert _metric in {"euclidean", "cosine", "random"}
            if _metric == "random":
                _metric = self.dist_for_rand
            _f_metric = euclidean if _metric == "euclidean" else cosine

            _nn_dist__per__class = {}
            for (
                _fclass,
                _nn_mean_feat__mat,
            ) in _res.nn_info.nn_mean_feat__mat__per__fclass.items():
                assert _nn_mean_feat__mat.shape == (
                    _res.nn_info.n_neighbors + 1,
                    _res.train_data_info.n_features,
                )
                _fclass__vec = _nn_mean_feat__mat[0]
                _nn_dist__per__class[_fclass] = mean(
                    _f_metric(_fclass__vec, _nn__vec)
                    for _nn__vec in _nn_mean_feat__mat[1:]
                )
            assert (
                set(_nn_dist__per__class.keys())
                == _res.train_data_info.class__set__per__split["few"]
            )

            _baseline_res = self._get_baseline_res(_dataset_name)
            _acc__per__class = _res.test_acc__per__class
            _baseline_acc__per__class = _baseline_res.test_acc__per__class

            # pylint: disable=consider-using-dict-items
            for _class in _nn_dist__per__class:
                df_rows.append(
                    {
                        "GID": _gid,
                        "LID": _lid,
                        "Experiment": _exp_name,
                        "Metric": _metric.title(),
                        "Dataset": SplitLTDataset(_dataset_name).proper_name,
                        "Class": _class,
                        "Test accuracy change": (
                            _acc__per__class[_class] - _baseline_acc__per__class[_class]
                        ),
                        "Mean NN distance": _nn_dist__per__class[_class],
                    }
                )
        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        assert len(df.groupby("Experiment")["Metric"].agg(set).values[0]) == 1
        assert len(df.groupby("Experiment")["Dataset"].agg(set).values[0]) == 1

        def annotate_with_corr(data, **kwargs):
            assert len(data["Experiment"].unique()) == 1
            _corr = (
                data[["Test accuracy change", "Mean NN distance"]].corr().values[0, 1]
            )
            _ax = plt.gca()
            _ax.text(0.75, 0.9, f"$r={_corr:.2f}$", transform=_ax.transAxes)

        self.plot.config()
        g = sns.lmplot(
            df,
            x="Test accuracy change",
            y="Mean NN distance",
            col="Experiment",
            col_wrap=self.col_wrap,
            markers=".",
            facet_kws=dict(sharex=False, sharey=False),
            scatter=True,
            fit_reg=True,
            units="GID",
            scatter_kws=dict(
                s=(mpl.rcParams["lines.markersize"] ** 2) / 2, alpha=0.5, linewidths=0
            ),
            n_boot=self.n_boot,
            **self._get_ha(),
        )
        g.map_dataframe(annotate_with_corr)
        g.set_titles("{col_name}")
        g.despine(left=True, right=True, top=True, bottom=True)
        for _ax in g.axes.flatten():
            _ax.grid(visible=True, axis="both", which="major")
            _ax.tick_params(axis="x", which="major", length=0)
        self._set_facet_grid_size(g)
        self._save_facet_grid(g)


class PlotSplitAccVsExp(_BaseMultiExpPlotCmd, BasePlotCmd):
    col: Literal["dataset", "metric"]
    y: Literal["acc", "acc_delta"]
    xlabel: str = ""
    legend_loc: str = "upper right"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None

    def __call__(self):
        df_rows = []
        for _gid, _lid, _exp_name, _dataset_name, _res in self._iter_train_results():
            _baseline_res = self._get_baseline_res(_dataset_name)
            for _split in ["Many", "Medium", "Few", "Overall"]:
                df_rows.append(
                    {
                        "GID": _gid,
                        "LID": _lid,
                        "Experiment": _exp_name,
                        "Dataset": SplitLTDataset(_dataset_name).proper_name,
                        "Metric": get_metric_desc(_res.nn_info.nn_dist),
                        "Split": _split,
                        "Accuracy": get_test_split_acc(_res, _split),
                        "Accuracy change": (
                            get_test_split_acc(_res, _split)
                            - get_test_split_acc(_baseline_res, _split)
                        ),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        if self.col == "Dataset":
            assert len(df["Metric"].unique()) == 1
        else:
            assert len(df["Dataset"].unique() == 1)

        self.plot.config()
        g = sns.catplot(
            df,
            x="Experiment",
            y=("Accuracy change" if self.y == "acc_delta" else "Accuracy"),
            hue="Split",
            hue_order=["Overall", "Few", "Medium", "Many"],
            col=self.col.title(),
            col_wrap=self.col_wrap,
            n_boot=self.n_boot,
            units="GID",
            kind="point",
            facet_kws=dict(sharex=False, sharey=True),
            dodge=0.4,
            markers=["x", ".", "d", "*"],
            **self._get_ha(),
        )
        g.refline(y=0, ls="-", color=self.plot.palette[0], zorder=1)
        if len(df[self.col.title()].unique()) > 1:
            g.set_titles("{col_name}")
        else:
            assert g.axes.shape == (1, 1)
            g.set_titles("")
        g.set_xlabels(self.xlabel)
        g.despine(left=True, right=True, bottom=False, top=True)
        if not self.legend_loc:
            g.legend.remove()
        else:
            sns.move_legend(
                g,
                self.legend_loc,
                ncols=4,
                title="",
                bbox_to_anchor=self.legend_bbox_to_anchor,
            )
        if self.y == "acc":
            g.set(ylim=(0, 1.01), yticks=[0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            g.set(ylim=(-0.25, 0.25), yticks=[-0.2, -0.1, 0, 0.1, 0.2])
        self._set_facet_grid_size(g)
        self._save_facet_grid(g)


class PlotPerClsAccVsSamples(_BaseMultiFilePlotCmd, BasePlotCmd):
    dataset: SplitLTDataset
    scatter: bool = False
    plot_params: PlotParams
    group_by_rhos: bool = True

    def __call__(self):
        baseline_res = load_train_res_with_per_cls_accs(
            self.dataset.baseline_eval_file_path, self.eval_batch_size
        )
        n_train_imgs__per__class = baseline_res.train_data_info.n_imgs__per__class

        df_rows = []
        rhos = set()
        for _id, (_res, _res_type) in enumerate(
            itertools.chain(
                ((_tres, "AlphaNet") for _tres in self._iter_train_results()),
                [(baseline_res, "Baseline")],
            )
        ):
            if _res.train_data_info.dataset_name != str(self.dataset):
                raise ValueError("result file does not match input dataset")

            if self.group_by_rhos:
                if (
                    _res_type == "AlphaNet"
                    and not _res.training_config.sampler_builder.sampler_classes
                    == (AllFewSampler, ClassBalancedBaseSampler)
                ):
                    raise ValueError(
                        f"bad sampler: {_res.training_config.sampler_builder}"
                    )

            _acc__per__class = _res.test_acc__per__class
            assert set(_acc__per__class.keys()) == set(n_train_imgs__per__class.keys())

            if _res_type == "AlphaNet":
                if self.group_by_rhos:
                    _rho = _res.training_config.sampler_builder.sampler_args[1]["r"]
                    rhos.add(_rho)
                    _exp_name = f"$\\rho={_rho}$"
                else:
                    _exp_name = f"AlphaNet-{_id}"
                    rhos.add(_id)
            else:
                _exp_name = "Baseline"

            for _class in _acc__per__class:
                df_rows.append(
                    {
                        "ID": _id,
                        "Experiment": _exp_name,
                        "Class": _class,
                        "Test accuracy": _acc__per__class[_class],
                        "Train samples": n_train_imgs__per__class[_class],
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        if self.group_by_rhos:
            _hue_order = [f"$\\rho={_rho}$" for _rho in sorted(rhos)]
        else:
            _hue_order = [f"AlphaNet-{_j}" for _j in sorted(rhos)]
        _hue_order = _hue_order + ["Baseline"]
        self.plot.config()
        g = sns.lmplot(
            data=df,
            x="Train samples",
            y="Test accuracy",
            hue="Experiment",
            units="ID",
            hue_order=_hue_order,
            logx=True,
            scatter=self.scatter,
            legend=False,
            palette=sns.color_palette("husl", n_colors=len(rhos)) + ["#000"],
            ci=None,
            scatter_kws=dict(s=(mpl.rcParams["lines.markersize"] / 2) ** 2),
            facet_kws=dict(despine=False, legend_out=False),
        )
        self.plot_params.set_params(g.ax)
        if len(_hue_order) > 4:
            g.add_legend(title="", ncols=2, bbox_to_anchor=(1, 1), loc="upper left")
        else:
            g.add_legend(title="", ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1))
        g.figure.set_size_inches(self.plot.get_size())
        if self.plot.file is not None:
            g.savefig(self.plot.file.name)


class PlotDeltaPerClassAccs(_BaseMultiFilePlotCmd, BasePlotCmd):
    dataset: SplitLTDataset

    def __call__(self):
        baseline_res = load_train_res_with_per_cls_accs(
            self.dataset.baseline_eval_file_path, self.eval_batch_size
        )

        def _get_split_for_class(_class):
            nonlocal baseline_res
            return next(
                (
                    _split
                    for _split, _split_classes in (
                        baseline_res.train_data_info.class__set__per__split.items()
                    )
                    if _class in _split_classes
                )
            )

        classes_ordered_by_size = sorted(
            baseline_res.train_data_info.n_imgs__per__class,
            key=lambda _c: baseline_res.train_data_info.n_imgs__per__class[_c],
        )

        df_rows = []
        overall_test_acc_deltas = set()
        test_acc_deltas__per__split = defaultdict(set)

        for _id, _res in enumerate(self._iter_train_results()):
            if _res.train_data_info.dataset_name != str(self.dataset):
                raise ValueError("result file does not match input dataset")

            _acc__per__class = _res.test_acc__per__class
            assert set(_acc__per__class.keys()) == set(
                baseline_res.test_acc__per__class.keys()
            )

            overall_test_acc_deltas.add(
                _res.test_metrics["accuracy"] - baseline_res.test_metrics["accuracy"]
            )
            for _split in ("many", "medium", "few"):
                test_acc_deltas__per__split[_split].add(
                    _res.test_acc__per__split[_split]
                    - baseline_res.test_acc__per__split[_split]
                )

            _delta_acc__per__class = {
                _class: _acc__per__class[_class]
                - baseline_res.test_acc__per__class[_class]
                for _class in _acc__per__class
            }

            for _index, _class in enumerate(classes_ordered_by_size):
                df_rows.append(
                    {
                        "ID": _id,
                        "Index": _index,
                        "Class": _class,
                        "Split": _get_split_for_class(_class),
                        "Delta test accuracy": _delta_acc__per__class[_class],
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded data frame:%s\n", df)

        with self.plot.open() as (_fig, _ax):
            _xticks = [0]
            for _split, _color in zip(("few", "medium", "many"), self.plot.palette[1:]):
                _split_df = df[df["Split"] == _split]
                sns.lineplot(
                    data=_split_df,
                    x="Index",
                    y="Delta test accuracy",
                    ax=_ax,
                    color=_color,
                    alpha=0.75,
                )

                _xticks.append(_split_df["Index"].max())
                _ax.hlines(
                    y=mean(test_acc_deltas__per__split[_split]),
                    xmin=_xticks[-2],
                    xmax=_xticks[-1],
                    colors=[_color],
                    linestyles=["--"],
                    zorder=-1,
                )
                _ax.text(
                    _split_df["Index"].mean(),
                    -0.7,
                    f"{_split.title()} split",
                    horizontalalignment="center",
                )

            _ax.set_xticks(_xticks[1:-1])
            _ax.set_xticklabels(["", ""])
            _ax.set_xlabel("")
            _ax.set_ylim(-0.6, 0.6)
            _ax.set_ylabel("Change in test accuracy")
            _ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
            _ax.axhline(y=0, ls="-", color=self.plot.palette[0])

            _mean_overall_acc_delta = mean(overall_test_acc_deltas)
            _ax.axhline(
                y=_mean_overall_acc_delta,
                ls="--",
                color=self.plot.palette[0],
                zorder=-1,
            )


class PlotAlphaDist(_BaseMultiFilePlotCmd, BasePlotCmd):
    plot_params: PlotParams
    show_xlabels: bool = False
    legend_loc: str = "lower right"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None
    col_wrap: Optional[int] = None

    def __call__(self):
        df_rows = []
        n_neighbors = None
        nn_dist = None

        for _id, _res in enumerate(self._iter_train_results()):
            if _id == 0:
                n_neighbors = _res.training_config.n_neighbors
                nn_dist = _res.training_config.nn_dist
            elif _res.training_config.n_neighbors != n_neighbors:
                raise ValueError("n_neighbors not same for result files")
            elif _res.training_config.nn_dist != nn_dist:
                raise ValueError("nn_dist not same for result files")

            _dataset_name = SplitLTDataset(
                _res.train_data_info.dataset_name
            ).proper_name
            _alphanet_classifier = _res.load_best_alphanet_classifier()
            _alphanet_classifier = _alphanet_classifier.to(DEFAULT_DEVICE).eval()
            _alpha__mat = _alphanet_classifier.get_learned_alpha_vecs()
            assert all(bool(_a0 == 1) for _a0 in _alpha__mat[:, 0])

            for (_target, _source) in itertools.product(
                range(_alpha__mat.shape[0]), range(1, _alpha__mat.shape[1])
            ):
                df_rows.append(
                    {
                        "ID": _id,
                        "Dataset": _dataset_name,
                        "Target": _target,
                        "Source": f"$\\alpha_{_source}$",
                        "Alpha": _alpha__mat[_target, _source].item(),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded data frame:\n%s", df)

        self.plot.config()
        _n_datasets = len(df["Dataset"].unique())
        _col_wrap = (
            _n_datasets // int(sqrt(_n_datasets))
            if self.col_wrap is None
            else self.col_wrap
        )
        _plot_size = self.plot.get_size()
        g = sns.displot(
            data=df,
            x="Alpha",
            hue="Source",
            col="Dataset",
            col_wrap=_col_wrap,
            col_order=sorted(list(df["Dataset"].unique())),
            kind="kde",
            legend=True,
            palette=itertools.cycle(self.plot.palette[1:]),
            # palette="husl",
            height=_plot_size[1],
            aspect=(_plot_size[0] / _plot_size[1]),
            facet_kws=dict(sharex=True, sharey=False),
            # alpha=0.5,
            # linewidth=0,
            # fill=True,
        )
        self.plot_params.set_params(g.axes)
        g.despine(top=True, bottom=False, left=True, right=True, trim=True)
        g.refline(x=0, ls="-", color=self.plot.palette[0], zorder=-1)
        g.set_titles("{col_name}")
        if not self.show_xlabels:
            g.set_xlabels("")
        g.set_ylabels("")
        for _ax in g.axes.flatten():
            _ax.tick_params(
                axis="x",
                which="major",
                reset=True,
                top=False,
                bottom=True,
                grid_linewidth=0,
            )
            _ax.set_xticklabels(list(map(str, _ax.get_xticks())))
        if self.legend_loc:
            sns.move_legend(
                g,
                loc=self.legend_loc,
                title="",
                bbox_to_anchor=self.legend_bbox_to_anchor,
            )
        else:
            g.legend.remove()
        _n_cols = _n_datasets / _col_wrap
        g.figure.set_size_inches(_plot_size[0] * _col_wrap, _plot_size[1] * _n_cols)
        if self.plot.file is not None:
            g.figure.savefig(
                self.plot.file, format=Path(self.plot.file.name).suffix[1:]
            )
            self.plot.file.close()


class PlotTemplateDeltas(BasePlotCmd):
    res_file: InputBinFile
    eval_batch_size: int = 1024
    delta_type: Literal["min", "max"] = "max"
    n_deltas: int = 1
    n_pcs: int = 50
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
        ) = get_per_class_test_accs(
            alphanet_res, self.eval_batch_size, return_preds=True
        )
        (
            baseline_test_acc__per__class,
            baseline_test_pred__seq,
        ) = get_per_class_test_accs(
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
        _mds = MDS(2, verbose=2, n_jobs=-1, normalized_stress="auto")
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
            ncols=2, nrows=self.n_deltas, sharex="row", sharey="row", squeeze=False
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


TEST_DATA_CACHE: Dict[str, SplitLTDataGroup] = {}


def get_per_class_test_accs(res: TrainResult, batch_size: int, return_preds=False):
    alphanet_classifier = res.load_best_alphanet_classifier()
    alphanet_classifier = alphanet_classifier.to(DEFAULT_DEVICE).eval()
    dataset = SplitLTDataset(res.train_data_info.dataset_name)
    try:
        test_datagrp = TEST_DATA_CACHE[res.train_data_info.dataset_name]
    except KeyError:
        test_datagrp = dataset.load_data(res.training_config.test_datagrp)
        TEST_DATA_CACHE[res.train_data_info.dataset_name] = test_datagrp
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

    logging.root.setLevel(logging.WARNING)
    eval_engine.run(test_data_loader)
    logging.root.setLevel(logging.INFO)
    _second_ret = eval_engine.state.my_yhat__seq if return_preds else None
    return eval_engine.state.my_accuracy__per__class, _second_ret
