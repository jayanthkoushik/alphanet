import itertools
import logging
from collections import defaultdict
from heapq import nlargest, nsmallest
from statistics import mean
from typing import Dict, Literal, Optional, Tuple

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
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from alphanet._dataset import SplitLTDataGroup, SplitLTDataset
from alphanet._samplers import AllFewSampler, ClassBalancedBaseSampler
from alphanet._utils import ContextPlot, CUD_PALETTE, PlotParams
from alphanet.train import TrainResult

logging.root.setLevel(logging.INFO)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePlotCmd(Corgy):
    def __call__(self):
        raise NotImplementedError


class _BaseMultiFilePlotCmd(Corgy):
    base_res_dir: InputDirectory
    res_sub_dirs: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    plot: ContextPlot

    def _iter_train_results(self):
        dataset_names = set()
        for _res_sub_dir in self.res_sub_dirs or [""]:
            for _res_file in tqdm(
                list((self.base_res_dir / _res_sub_dir).glob(self.res_files_pattern)),
                desc=f"Loading {_res_sub_dir}".rstrip(),
                unit="file",
            ):
                res = TrainResult.from_dict(
                    torch.load(_res_file, map_location=DEFAULT_DEVICE)
                )
                dataset_names.add(res.train_data_info.dataset_name)
                yield res


class PlotPerClsAccVsSamples(_BaseMultiFilePlotCmd, BasePlotCmd):
    dataset: SplitLTDataset
    eval_batch_size: int = 1024
    scatter: bool = False
    plot_params: PlotParams

    def __call__(self):
        baseline_res = TrainResult.from_dict(
            torch.load(
                self.dataset.baseline_eval_file_path, map_location=DEFAULT_DEVICE
            )
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

            if (
                _res_type == "AlphaNet"
                and not _res.training_config.sampler_builder.sampler_classes
                == (AllFewSampler, ClassBalancedBaseSampler)
            ):
                raise ValueError(f"bad sampler: {_res.training_config.sampler_builder}")

            _acc__per__class = _res.test_acc__per__class
            if _acc__per__class is None:
                tqdm.write(
                    f"generating per-class test accuracies for {_res_type.lower()} file"
                )
                _acc__per__class, _ = get_per_class_test_accs(
                    _res, self.eval_batch_size
                )
            assert set(_acc__per__class.keys()) == set(n_train_imgs__per__class.keys())

            if _res_type == "AlphaNet":
                _rho = _res.training_config.sampler_builder.sampler_args[1]["r"]
                rhos.add(_rho)
                _exp_name = f"$\\rho={_rho}$"
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

        _hue_order = ["Baseline"] + [f"$\\rho={_rho}$" for _rho in sorted(rhos)]
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
            facet_kws=dict(despine=False, legend_out=False),
        )
        self.plot_params.set_params(g.ax)
        g.add_legend(title="", loc="upper left", bbox_to_anchor=(0.1, 1))
        g.figure.set_size_inches(self.plot.get_size())
        if self.plot.file is not None:
            g.savefig(self.plot.file.name)


class PlotDeltaPerClassAccs(_BaseMultiFilePlotCmd, BasePlotCmd):
    dataset: SplitLTDataset
    eval_batch_size: int = 1024

    def __call__(self):
        baseline_res = TrainResult.from_dict(
            torch.load(
                self.dataset.baseline_eval_file_path, map_location=DEFAULT_DEVICE
            )
        )
        baseline_test_acc__per__class = baseline_res.test_acc__per__class
        if baseline_test_acc__per__class is None:
            tqdm.write("generating per-class test accuracies for baseline")
            baseline_test_acc__per__class, _ = get_per_class_test_accs(
                baseline_res, self.eval_batch_size
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
            if _acc__per__class is None:
                tqdm.write("generating per-class test accuracies for file")
                _acc__per__class, _ = get_per_class_test_accs(
                    _res, self.eval_batch_size
                )
            assert set(_acc__per__class.keys()) == set(
                baseline_test_acc__per__class.keys()
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
                _class: _acc__per__class[_class] - baseline_test_acc__per__class[_class]
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
            for _split, _color in zip(("few", "medium", "many"), CUD_PALETTE[1:]):
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
            _ax.axhline(y=0, ls="-", color="k")

            _mean_overall_acc_delta = mean(overall_test_acc_deltas)
            _ax.axhline(y=_mean_overall_acc_delta, ls="--", color="k", zorder=-1)


class PlotAlphaDist(_BaseMultiFilePlotCmd, BasePlotCmd):
    plot_params: PlotParams

    def __call__(self):
        df_rows = []
        dataset_name = None
        n_neighbors = None
        nn_dist = None

        for _id, _res in enumerate(self._iter_train_results()):
            if _id == 0:
                dataset_name = _res.train_data_info.dataset_name
                n_neighbors = _res.training_config.n_neighbors
                nn_dist = _res.training_config.nn_dist
            elif _res.train_data_info.dataset_name != dataset_name:
                raise ValueError("dataset not same for result files")
            elif _res.training_config.n_neighbors != n_neighbors:
                raise ValueError("n_neighbors not same for result files")
            elif _res.training_config.nn_dist != nn_dist:
                raise ValueError("nn_dist not same for result files")

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
                        "Target": _target,
                        "Source": f"$\\alpha_{_source}$",
                        "Alpha": _alpha__mat[_target, _source].item(),
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded data frame:\n%s", df)

        self.plot.config()
        g = sns.displot(
            data=df,
            x="Alpha",
            hue="Source",
            kind="kde",
            legend=True,
            palette=itertools.cycle(CUD_PALETTE[1:]),
            facet_kws=dict(despine=False, legend_out=False),
        )
        self.plot_params.set_params(g.ax)
        g.ax.set_xlabel("$\\alpha$")
        g.legend.set_title("")
        g.despine(top=True, bottom=False, left=True, right=True, trim=True)
        g.figure.set_size_inches(self.plot.get_size())
        if self.plot.file is not None:
            g.savefig(self.plot.file)
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
        if alphanet_res.training_config.n_neighbors > len(CUD_PALETTE) - 1:
            raise ValueError(
                f"cannot handle results with more than "
                f"'{len(CUD_PALETTE)}' nearest neighbors"
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
                        edgecolors="k",
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
                            edgecolors="k",
                            marker="*",
                            zorder=1,
                        )

                    # Plot the test projection embeddings.
                    for _test_proj__vec, _test_pred in zip(
                        _test_proj__mat, _test_pred__seq
                    ):
                        assert _test_proj__vec.shape == (2,)
                        if _test_pred == _fclass:
                            _color = "k"
                        else:
                            try:
                                _idx = (
                                    alphanet_res.nn_info.nn_class__seq__per__fclass[
                                        _fclass
                                    ]
                                ).index(_test_pred)
                                _color = CUD_PALETTE[1 + _idx]
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
                        _edgecolors = CUD_PALETTE[1 + _j]
                        _color = _edgecolors + "bb"
                        _dot_rad = _nn_dot_rad__seq[_j]
                        _ax.scatter(
                            [_nn_embed__vec[0].item()],
                            [_nn_embed__vec[1].item()],
                            s=(_dot_rad**2),
                            c=_color,
                            edgecolors="k",
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

    eval_engine.run(test_data_loader)
    _second_ret = eval_engine.state.my_yhat__seq if return_preds else None
    return eval_engine.state.my_accuracy__per__class, _second_ret
