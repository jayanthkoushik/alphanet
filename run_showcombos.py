#!/usr/bin/env python3

from argparse import SUPPRESS
from collections import defaultdict
from heapq import nlargest, nsmallest
from statistics import mean
from typing import Dict, List, Literal, Tuple

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

        _alpha__mat__per__fclass = {
            _fclass: torch.stack(_alpha__vec__seq)
            for _fclass, _alpha__vec__seq in alpha__vec__seq__per__fclass.items()
        }
        self.mean_std_alpha__vec__per__fclass = {
            _fclass: (_alpha__mat.mean(dim=0), _alpha__mat.std(dim=0))
            for _fclass, _alpha__mat in _alpha__mat__per__fclass.items()
        }

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
            f"({_alpha_means[_j]:.1f}±{_alpha_stds[_j]:.2g})[{_nn_class_labels[_j]}]"
            for _j in range(len(_nn_class_labels))
        )
        return s


class Main(Corgy):
    base_res_dir: InputDirectory
    res_files_pattern: str = "**/*.pth"
    select_by: Literal["acc", "acc_delta"] = "acc_delta"
    select_max: bool = True
    select_n: int = 5
    eval_batch_size: int = 1024

    def __call__(self) -> None:
        dataset_name: str
        alpha__vec__seq__per__fclass: Dict[int, List[torch.Tensor]] = defaultdict(list)
        test_acc__seq__per__class: Dict[int, List[float]] = defaultdict(list)
        n_mistakes__per__class__fclass: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        n_preds__per__fclass: Dict[int, int] = defaultdict(int)

        result_files = list(self.base_res_dir.glob(self.res_files_pattern))
        for _i, _res_file in enumerate(tqdm(result_files, desc="Loading", unit="file")):
            _res = TrainResult.from_dict(torch.load(_res_file, map_location=DEVICE))

            if _i == 0:
                dataset_name = _res.train_data_info.dataset_name
            elif dataset_name != _res.train_data_info.dataset_name:
                raise ValueError("all result files must be from same dataset")

            _test_acc__per__class, _test_yhats = _get_test_acc_per_class(
                _res, self.eval_batch_size, return_preds=True
            )
            assert _test_yhats is not None
            for _class, _class_acc in _test_acc__per__class.items():
                test_acc__seq__per__class[_class].append(_class_acc)

            _test_ys = _TEST_DATA_CACHE[dataset_name].label__seq
            for _y, _yhat in zip(_test_ys, _test_yhats):
                if _y in _res.train_data_info.class__set__per__split["few"]:
                    n_preds__per__fclass[_y] += 1
                    if _y != _yhat:
                        n_mistakes__per__class__fclass[_y][_yhat] += 1

            _alphanet_classifier = _res.load_best_alphanet_classifier()
            _alphanet_classifier = _alphanet_classifier.to(DEVICE).eval()
            _alphas = _alphanet_classifier.get_learned_alpha_vecs()
            assert _alphas.shape[1] == _res.nn_info.n_neighbors + 1
            assert torch.allclose(_alphas[:, 0], torch.ones(_alphas.shape[0]))
            for _class, _idx in enumerate(_res.fbclass_ordered_idx__vec):
                if _class in _res.train_data_info.class__set__per__split["few"]:
                    assert _idx < _res.train_data_info.n_few_classes
                    alpha__vec__seq__per__fclass[_class].append(_alphas[_idx])

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
        baseline_test_acc__per__class = baseline_res.test_acc__per__class
        assert baseline_test_acc__per__class is not None

        mean_test_acc__per__class: Dict[int, float] = {
            _class: mean(_test_acc__seq)
            for _class, _test_acc__seq in test_acc__seq__per__class.items()
        }
        mean_test_acc_delta__per__class = {
            _class: mean_test_acc__per__class[_class]
            - baseline_test_acc__per__class[_class]
            for _class in mean_test_acc__per__class
        }

        combo_builder = ComboBuilder(
            label_name__per__class,
            baseline_res.nn_info.nn_class__seq__per__fclass,
            alpha__vec__seq__per__fclass,
        )

        _f_select = nlargest if self.select_max else nsmallest
        _key_dic = (
            mean_test_acc__per__class
            if self.select_by == "acc"
            else mean_test_acc_delta__per__class
        )
        selected_fclasses = _f_select(
            self.select_n,
            baseline_res.train_data_info.class__set__per__split["few"],
            lambda _c: _key_dic[_c],
        )

        for _fclass in selected_fclasses:
            print("*********")
            print(combo_builder.get_combo_str(_fclass))
            print(f"Training samples: {train_samples__per__class[_fclass]}")
            print(f"Baseline accuracy: {baseline_test_acc__per__class[_fclass]:.2f}")
            print(f"Mean AlphaNet accuracy: {mean_test_acc__per__class[_fclass]:.2f}")

            _top_misclassifications = nlargest(
                self.select_n,
                n_mistakes__per__class__fclass[_fclass],
                # pylint: disable=cell-var-from-loop
                key=lambda _c: n_mistakes__per__class__fclass[_fclass][_c],
            )
            _n_misclassifications = sum(
                n_mistakes__per__class__fclass[_fclass].values()
            )
            print(
                f"Top misclassifications ("
                f"{_n_misclassifications}/{n_preds__per__fclass[_fclass]} across "
                f"{len(result_files)} repetitions):"
            )
            for _class in _top_misclassifications:
                _split = next(
                    _s
                    for _s in ("few", "medium", "many")
                    if _class in baseline_res.train_data_info.class__set__per__split[_s]
                )
                _label = label_name__per__class[_class]
                print(
                    f"\t{_label} ('{_split}' split): "
                    f"{n_mistakes__per__class__fclass[_fclass][_class]}"
                )
            print("*********\n")


if __name__ == "__main__":
    Main.parse_from_cmdline(usage=SUPPRESS)()
