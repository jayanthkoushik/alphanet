import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

import ignite.contrib.handlers
import ignite.engine
import ignite.metrics
import pandas as pd
import seaborn as sns
import torch
from corgy import Corgy
from corgy.types import InputDirectory
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from alphanet._dataset import SplitLTDataset
from alphanet._samplers import AllFewSampler, ClassBalancedBaseSampler
from alphanet._utils import ContextPlot
from alphanet.train import TrainResult

logging.root.setLevel(logging.INFO)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _PlotBase(Corgy):
    base_res_dir: InputDirectory
    res_sub_dirs: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    plot: ContextPlot

    def _iter_train_results(self, include_baselines=False):
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
                yield res, "AlphaNet"

        for _dataset_name in dataset_names:
            _dataset = SplitLTDataset(_dataset_name)
            res = TrainResult.from_dict(
                torch.load(
                    _dataset.baseline_eval_file_path, map_location=DEFAULT_DEVICE
                )
            )
            yield res, "Baseline"


class PlotPerClsAccVsSamples(_PlotBase):
    eval_batch_size: int = 1024

    def _get_per_class_test_accs(self, res: TrainResult):
        alphanet_classifier = res.load_best_alphanet_classifier()
        dataset = SplitLTDataset(res.train_data_info.dataset_name)
        test_datagrp = dataset.load_data("test")
        test_data_loader = DataLoader(
            TensorDataset(
                test_datagrp.feat__mat, torch.tensor(test_datagrp.label__seq)
            ),
            self.eval_batch_size,
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
                    _correct_preds__per__class[_class]
                    / _total_preds__per__class[_class]
                )
                for _class in _correct_preds__per__class
            }

            assert all(
                _total_preds__per__class[_class] == _class_imgs
                for _class, _class_imgs in test_datagrp.info.n_imgs__per__class.items()
            )
            assert torch.isclose(
                torch.tensor(
                    [sum(_correct_preds__per__class.values())], dtype=torch.float
                ),
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

        eval_engine.run(test_data_loader)
        return eval_engine.state.my_accuracy__per__class

    def __call__(self):
        df_rows = []
        dataset_name = None
        rhos = set()
        for _id, (_res, _res_type) in enumerate(
            self._iter_train_results(include_baselines=True)
        ):
            if not _res.training_config.sampler_builder.sampler_classes == (
                AllFewSampler,
                ClassBalancedBaseSampler,
            ):
                raise ValueError(f"bad sampler: {_res.training_config.sampler_builder}")
            if dataset_name is None:
                dataset_name = _res.train_data_info.dataset_name
            elif dataset_name != _res.train_data_info.dataset_name:
                raise ValueError("all results not from same dataset")

            _acc__per__class = self._get_per_class_test_accs(_res)
            _n_imgs__per__class = _res.train_data_info.n_imgs__per__class
            assert set(_acc__per__class.keys()) == set(_n_imgs__per__class.keys())
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
                        "Train samples": _n_imgs__per__class[_class],
                    }
                )

        df = pd.DataFrame(df_rows)
        logging.info("loaded dataframe:\n%s", df)

        self.plot.config()
        _hue_order = ["Baseline"] + [f"$\\rho={_rho}$" for _rho in sorted(rhos)]
        g = sns.lmplot(
            data=df,
            x="Train samples",
            y="Test accuracy",
            hue="Experiment",
            units="ID",
            hue_order=_hue_order,
            logx=True,
            scatter=False,
            legend=False,
            facet_kws=dict(despine=False, legend_out=False),
        )
        g.ax.set_xscale("log")
        g.ax.set_xlim(1, 1000)
        g.ax.set_ylim(0, 1.01)
        g.ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        g.add_legend(title="", loc="upper left", bbox_to_anchor=(0.1, 1))
        g.figure.set_size_inches(self.plot.get_size())
        if self.plot.file is not None:
            g.savefig(self.plot.file)
            self.plot.file.close()
