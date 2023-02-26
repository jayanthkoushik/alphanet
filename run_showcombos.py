#!/usr/bin/env python3

from argparse import SUPPRESS
from collections import defaultdict
from heapq import nlargest, nsmallest
from math import isnan
from statistics import mean
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch
from corgy import Corgy
from corgy.types import InputDirectory
from tqdm import tqdm

from alphanet._dataset import SplitLTDataset
from alphanet.plot import _get_test_acc_per_class, _TEST_DATA_CACHE
from alphanet.train import TrainResult

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComboBuilder:
    label_name__per__class: Dict[int, str]
    nn_class__seq__per__fclass: Dict[int, List[int]]
    mean_std_alpha__vec__per__fclass: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __init__(
        self,
        label_name__per__class: Dict[int, str],
        nn_class__seq__per__fclass: Dict[int, List[int]],
        alpha__vec__seq__per__fclass: Dict[int, List[torch.Tensor]],
    ):
        self.label_name__per__class = label_name__per__class
        self.nn_class__seq__per__fclass = nn_class__seq__per__fclass

        assert all(
            len(nn_class__seq__per__fclass[_fclass])
            == len(alpha__vec__seq__per__fclass[_fclass][0])
            for _fclass in nn_class__seq__per__fclass
        )

        _alpha__mat__per__fclass = {
            _fclass: torch.stack(_alpha__vec__seq)
            for _fclass, _alpha__vec__seq in alpha__vec__seq__per__fclass.items()
        }
        self.mean_std_alpha__vec__per__fclass = {
            _fclass: (_alpha__mat.mean(dim=0), _alpha__mat.std(dim=0))
            for _fclass, _alpha__mat in _alpha__mat__per__fclass.items()
        }

    @staticmethod
    def _fmt_mean_sig(mu, sig):
        s = f"({mu:.2g}"
        if isnan(sig):
            return s + ")"
        return s + f"±{sig:.2g})"

    def get_combo_str(self, fclass: int) -> str:
        _alpha_means, _alpha_stds = [
            _vec.tolist() for _vec in self.mean_std_alpha__vec__per__fclass[fclass]
        ]
        _fclass_label = self.label_name__per__class[fclass]
        _nn_class_labels = [
            self.label_name__per__class[_nn_class]
            for _nn_class in self.nn_class__seq__per__fclass[fclass]
        ]

        _leading_pad = " " * (len(_fclass_label) + 5)
        s = f"[{_fclass_label}] = [{_fclass_label}]\n{_leading_pad}+ "
        s += f"\n{_leading_pad}+ ".join(
            (
                self._fmt_mean_sig(_alpha_means[_j], _alpha_stds[_j])
                + f"[{_nn_class_labels[_j]}]"
            )
            for _j in range(len(_nn_class_labels))
        )
        return s


class Main(Corgy):
    base_res_dir: InputDirectory
    res_files_pattern: str = "**/*.pth"
    select_by: Literal["acc", "acc_delta"] = "acc_delta"
    select_max: bool = True
    select_n: int = 5
    mistake_n: Optional[int] = None
    eval_batch_size: int = 1024

    @staticmethod
    def _process_mistakes(ys, yhats, n_mistakes_dic, n_preds_dic):
        for _y, _yhat in zip(ys, yhats):
            n_preds_dic[_y] += 1
            if _y != _yhat:
                n_mistakes_dic[_y][_yhat] += 1

    def __call__(self) -> None:
        dataset_name: str
        alpha__vec__seq__per__fclass: Dict[int, List[torch.Tensor]] = defaultdict(list)
        test_post_acc__seq__per__class: Dict[int, List[float]] = defaultdict(list)
        n_post_mistakes__per__class__class: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        n_post_preds__per__class: Dict[int, int] = defaultdict(int)
        fclass__set: Set[int]
        test_ys: List[int]
        nn_class__seq__per__fclass: Dict[int, List[int]]

        result_files = list(self.base_res_dir.glob(self.res_files_pattern))
        for _i, _res_file in enumerate(tqdm(result_files, desc="Loading", unit="file")):
            _res = TrainResult.from_dict(torch.load(_res_file, map_location=DEVICE))

            if _i == 0:
                dataset_name = _res.train_data_info.dataset_name
                fclass__set = _res.train_data_info.class__set__per__split["few"]
            elif dataset_name != _res.train_data_info.dataset_name:
                raise ValueError("all result files must be from same dataset")

            _test_acc__per__class, _test_yhats = _get_test_acc_per_class(
                _res, self.eval_batch_size, return_preds=True
            )
            if _i == 0:
                test_ys = _TEST_DATA_CACHE[dataset_name].label__seq
                nn_class__seq__per__fclass = _res.nn_info.nn_class__seq__per__fclass
            else:
                assert all(
                    nn_class__seq__per__fclass[_fclass]
                    == _res.nn_info.nn_class__seq__per__fclass[_fclass]
                    for _fclass in nn_class__seq__per__fclass
                )

            for _class, _class_acc in _test_acc__per__class.items():
                test_post_acc__seq__per__class[_class].append(_class_acc)

            assert _test_yhats is not None
            self._process_mistakes(
                test_ys,
                _test_yhats,
                n_post_mistakes__per__class__class,
                n_post_preds__per__class,
            )

            _alphanet_classifier = _res.load_best_alphanet_classifier()
            _alphanet_classifier = _alphanet_classifier.to(DEVICE).eval()
            _alphas = _alphanet_classifier.get_learned_alpha_vecs()
            assert _alphas.shape[1] == _res.nn_info.n_neighbors + 1
            assert torch.allclose(
                _alphas[:, 0], torch.ones(_alphas.shape[0], device=DEVICE)
            )
            for _class, _idx in enumerate(_res.fbclass_ordered_idx__vec):
                if _class in fclass__set:
                    assert _idx < len(fclass__set)
                    alpha__vec__seq__per__fclass[_class].append(_alphas[_idx][1:])

        dataset = SplitLTDataset(dataset_name)
        if dataset.label_names_file_path is None:
            raise ValueError("no label names for dataset")
        with open(dataset.label_names_file_path, "r", encoding="utf-8") as _f:
            label_name__per__class = {
                _i: _line.strip() for (_i, _line) in enumerate(_f) if _line.strip()
            }

        baseline_res = TrainResult.from_dict(
            torch.load(dataset.baseline_eval_file_path, map_location=DEVICE)
        )
        assert len(label_name__per__class) == baseline_res.train_data_info.n_classes
        train_samples__per__class = baseline_res.train_data_info.n_imgs__per__class
        test_pre_acc__per__class, baseline_test_yhats = _get_test_acc_per_class(
            baseline_res, self.eval_batch_size, return_preds=True
        )

        n_pre_mistakes__per__class__class: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        n_pre_preds__per__class: Dict[int, int] = defaultdict(int)
        assert baseline_test_yhats is not None
        self._process_mistakes(
            test_ys,
            baseline_test_yhats,
            n_pre_mistakes__per__class__class,
            n_pre_preds__per__class,
        )

        assert all(
            n_pre_preds__per__class[_class] * len(result_files)
            == n_post_preds__per__class[_class]
            for _class in n_pre_preds__per__class
        )

        mean_test_post_acc__per__class: Dict[int, float] = {
            _class: mean(_test_acc__seq)
            for _class, _test_acc__seq in test_post_acc__seq__per__class.items()
        }
        mean_test_acc_delta__per__class = {
            _class: mean_test_post_acc__per__class[_class]
            - test_pre_acc__per__class[_class]
            for _class in mean_test_post_acc__per__class
        }

        combo_builder = ComboBuilder(
            label_name__per__class,
            nn_class__seq__per__fclass,
            alpha__vec__seq__per__fclass,
        )

        _f_select = nlargest if self.select_max else nsmallest
        _key_dic = (
            mean_test_post_acc__per__class
            if self.select_by == "acc"
            else mean_test_acc_delta__per__class
        )
        selected_fclasses = _f_select(
            self.select_n, fclass__set, lambda _c: _key_dic[_c]
        )

        for _sel_fclass in selected_fclasses:
            print("********************************************")
            print(combo_builder.get_combo_str(_sel_fclass))
            print(f"\nTraining samples: {train_samples__per__class[_sel_fclass]}")
            print(f"Baseline accuracy: {test_pre_acc__per__class[_sel_fclass]:.2f}")
            print(
                f"Mean AlphaNet accuracy: "
                f"{mean_test_post_acc__per__class[_sel_fclass]:.2f}\n"
            )

            _mistake_n = (
                self.mistake_n
                if self.mistake_n is not None
                else len(n_pre_mistakes__per__class__class[_sel_fclass])
            )
            _top_pre_misclassifications = nlargest(
                _mistake_n,
                n_pre_mistakes__per__class__class[_sel_fclass],
                # pylint: disable=cell-var-from-loop
                key=lambda _c: n_pre_mistakes__per__class__class[_sel_fclass][_c],
            )
            _n_total_pre_misclassifications = sum(
                n_pre_mistakes__per__class__class[_sel_fclass].values()
            )
            __z = 1.0 - (
                _n_total_pre_misclassifications / n_pre_preds__per__class[_sel_fclass]
            )
            assert torch.isclose(
                torch.tensor(data=__z),
                torch.tensor(data=test_pre_acc__per__class[_sel_fclass]),
            )

            _mistake_n = (
                self.mistake_n
                if self.mistake_n is not None
                else len(n_post_mistakes__per__class__class[_sel_fclass])
            )
            _top_post_misclassifications = nlargest(
                _mistake_n,
                n_post_mistakes__per__class__class[_sel_fclass],
                # pylint: disable=cell-var-from-loop
                key=lambda _c: n_post_mistakes__per__class__class[_sel_fclass][_c],
            )
            _n_total_post_misclassifications = sum(
                n_post_mistakes__per__class__class[_sel_fclass].values()
            )
            __z = 1.0 - (
                _n_total_post_misclassifications / n_post_preds__per__class[_sel_fclass]
            )
            assert torch.isclose(
                torch.tensor(data=__z),
                torch.tensor(data=mean_test_post_acc__per__class[_sel_fclass]),
            )

            def _print_mistakes(_fclass, _mistakes):
                for c in _mistakes:
                    _split = next(
                        _s
                        for _s in ("few", "medium", "many")
                        if c in baseline_res.train_data_info.class__set__per__split[_s]
                    )
                    _label = label_name__per__class[c]
                    _n_pre_mistakes = n_pre_mistakes__per__class__class[_fclass][c]
                    _n_post_mistakes = n_post_mistakes__per__class__class[_fclass][c]
                    _mean_n_post_mistakes = _n_post_mistakes / len(result_files)
                    print(
                        f"\t{_label} ('{_split}' split): "
                        f"{_n_pre_mistakes} -> {_mean_n_post_mistakes:.2f} "
                        f"({_n_post_mistakes}/{len(result_files)})"
                    )

                    _n_cs_pre_errs = n_pre_mistakes__per__class__class[c][_fclass]
                    _n_cs_post_errs = n_post_mistakes__per__class__class[c][_fclass]
                    _mean_n_cs_post_errs = _n_cs_post_errs / len(result_files)
                    print(
                        f"\t\t|-> as '{label_name__per__class[_fclass]}': "
                        f"{_n_cs_pre_errs} -> {_mean_n_cs_post_errs:.2f} "
                        f"({_n_cs_post_errs}/{len(result_files)})"
                    )

            print(
                f"Top baseline misclassifications (total "
                f"{_n_total_pre_misclassifications} mistakes / "
                f"{n_pre_preds__per__class[_sel_fclass]} predictions),\n"
                f"and corresponding mean AlphaNet misclassifications\n"
                f"(total {_n_total_post_misclassifications} mistakes / "
                f"{n_post_preds__per__class[_sel_fclass]} predictions, across "
                f"{len(result_files)} repetition(s)):"
            )
            _print_mistakes(_sel_fclass, _top_pre_misclassifications)

            print("\nTop AlphaNet misclassifications:")
            _print_mistakes(_sel_fclass, _top_post_misclassifications)
            print("********************************************\n")


if __name__ == "__main__":
    Main.parse_from_cmdline(usage=SUPPRESS)()
