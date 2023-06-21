#!/usr/bin/env python3

from argparse import SUPPRESS
from collections import defaultdict
from math import isnan
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from corgy import Corgy
from corgy.types import InputDirectory
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import MulticlassAccuracy
from tqdm import trange

from alphanet._dataset import SplitLTDataset
from alphanet._pt import DEFAULT_DEVICE
from alphanet._wordnet import get_wordnet_nns_per_imgnet_class
from alphanet.plot import (
    _get_nn_dist_per_split_class,
    _get_test_acc_per_class,
    _TEST_DATA_CACHE,
)
from alphanet.train import TrainResult


class Args(Corgy):
    base_res_dir: InputDirectory
    rel_exp_paths: Tuple[Path]
    exp_names: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    exp_prefix: str = ""
    exp_suffix: str = ""
    acc_k: int = 1
    show_adjusted_acc: bool = False
    adjusted_acc_semantic: bool = False
    adjusted_acc_semantic_nns_level: Optional[int] = None
    imagenet_data_root: Optional[Path] = None
    eval_batch_size: int = 1024
    print_csv: bool = True


args = Args.parse_from_cmdline(usage=SUPPRESS)
table_rows: List[Dict[str, Any]] = []
max_exp_name_len = len("Experiment")
seen_datasets = set()


if args.show_adjusted_acc and args.adjusted_acc_semantic:
    assert args.adjusted_acc_semantic_nns_level is not None
    assert args.imagenet_data_root is not None
    wordnet_nn__seq__per__split_class = get_wordnet_nns_per_imgnet_class(
        args.adjusted_acc_semantic_nns_level,
        "all",
        str(args.imagenet_data_root / "splits" / "few.txt"),
        str(args.imagenet_data_root / "label_names.txt"),
        str(args.imagenet_data_root / "labels_full.txt"),
    )


def _get_adjusted_accs(_train_res: TrainResult, _args: Args) -> Dict[str, float]:
    _dataset = SplitLTDataset(_train_res.train_data_info.dataset_name)
    _, _res_test_yhats = _get_test_acc_per_class(
        _train_res, _args.eval_batch_size, return_preds=True
    )
    _test_datagrp = _TEST_DATA_CACHE[str(_dataset)]
    _test_ys = _test_datagrp.label__seq

    if args.adjusted_acc_semantic:
        _nn__seq__per__split_class = wordnet_nn__seq__per__split_class
    else:
        _, _nn__seq__per__split_class = _get_nn_dist_per_split_class(
            _dataset,
            _train_res.nn_info.nn_dist,
            _train_res.nn_info.n_neighbors,
            for_split="all",
            against_split="all",
        )

    _correct__per__split: Dict[str, int] = defaultdict(int)
    _total__per__split: Dict[str, int] = defaultdict(int)
    assert _res_test_yhats is not None
    for _y, _res_yhat in zip(_test_ys, _res_test_yhats):
        for (
            _split,
            _split_class__set,
        ) in _train_res.train_data_info.class__set__per__split.items():
            if _y in _split_class__set:
                break
        else:
            raise AssertionError

        if _res_yhat == _y or _res_yhat in _nn__seq__per__split_class[_y]:
            _correct__per__split[_split] += 1
        _total__per__split[_split] += 1
    assert sum(_total__per__split.values()) == len(_test_ys)

    _res_dict = {}
    for _split in _correct__per__split:
        _res_dict[_split.title()] = (
            _correct__per__split[_split] / _total__per__split[_split] * 100
        )
    _res_dict["Overall"] = sum(_correct__per__split.values()) / len(_test_ys) * 100
    return _res_dict


def _get_topk_acc(_train_res: TrainResult, _args: Args):
    _alphanet_classifier = _train_res.load_best_alphanet_classifier()
    _dataset = SplitLTDataset(_train_res.train_data_info.dataset_name)
    try:
        _test_datagrp = _TEST_DATA_CACHE[str(_dataset)]
    except KeyError:
        _test_datagrp = _dataset.load_data("test")
        _TEST_DATA_CACHE[str(_dataset)] = _test_datagrp
    _test_dataset = TensorDataset(
        _test_datagrp.feat__mat, torch.tensor(_test_datagrp.label__seq)
    )
    _data_loader = DataLoader(_test_dataset, args.eval_batch_size)

    _topk_metric = MulticlassAccuracy(k=args.acc_k)
    _topk_metric__per__split = {
        _split: MulticlassAccuracy(k=args.acc_k) for _split in ["many", "medium", "few"]
    }

    _pbar = trange(
        len(_test_dataset), desc=f"Computing top-{args.acc_k} accuracy", unit="sample"
    )
    for _X_batch, _y_batch in _data_loader:
        _X_batch, _y_batch = _X_batch.to(DEFAULT_DEVICE), _y_batch.to(DEFAULT_DEVICE)
        _yhat_batch = _alphanet_classifier(_X_batch)
        _topk_metric.update(_yhat_batch, _y_batch)
        for _yhat_i, _yi in zip(_yhat_batch, _y_batch):
            for (
                _split,
                _split_class__set,
            ) in _train_res.train_data_info.class__set__per__split.items():
                if int(_yi) in _split_class__set:
                    break
            else:
                raise AssertionError
            _topk_metric__per__split[_split].update(
                _yhat_i.unsqueeze(0), _yi.unsqueeze(0)
            )
        _pbar.update(len(_X_batch))
    _pbar.close()

    _res_dict = {"Overall": _topk_metric.compute() * 100}
    for _split, _split_topk_metric in _topk_metric__per__split.items():
        _res_dict[_split.title()] = _split_topk_metric.compute() * 100
    return _res_dict


pbar = trange(len(args.rel_exp_paths), desc="Loading", unit="experiment")
for _i, _rel_exp_path in enumerate(args.rel_exp_paths):
    _exp_res_dir = args.base_res_dir / _rel_exp_path
    # pylint: disable=unsubscriptable-object
    _exp_name = str(_rel_exp_path) if args.exp_names is None else args.exp_names[_i]
    _exp_name = args.exp_prefix + _exp_name + args.exp_suffix
    if (_exp_name_len := len(_exp_name)) > max_exp_name_len:
        max_exp_name_len = _exp_name_len

    if not _exp_res_dir.exists():
        raise ValueError(f"missing results folder: {_exp_res_dir}")

    _sub_pbar = trange(
        len(list(_exp_res_dir.glob(args.res_files_pattern))),
        desc=f"Experiment {_exp_name}",
        unit="file",
    )
    for _res_file in _exp_res_dir.glob(args.res_files_pattern):
        _res = TrainResult.from_dict(
            torch.load(str(_res_file), map_location=DEFAULT_DEVICE)
        )
        seen_datasets.add(_res.train_data_info.dataset_name)
        _res_dict = {
            "Experiment": _exp_name,
            "Dataset": _res.train_data_info.dataset_name,
        }
        if not args.show_adjusted_acc:
            if args.acc_k != 1:
                _res_dict.update(_get_topk_acc(_res, args))
            else:
                for _split, _split_acc in _res.test_acc__per__split.items():
                    _res_dict[_split.title()] = _split_acc * 100
                _res_dict["Overall"] = _res.test_metrics["accuracy"] * 100
                # for _split, _split_acc in _get_topk_acc(_res, args).items():
                #     assert torch.isclose(
                #         torch.tensor(data=_res_dict[_split]), _split_acc
                #     )
        else:
            _res_dict.update(_get_adjusted_accs(_res, args))

        table_rows.append(_res_dict)
        _sub_pbar.update(1)
    _sub_pbar.close()

    pbar.update(1)
pbar.close()

pbar = trange(len(seen_datasets), desc="Loading baselines", unit="dataset")
for _dataset_name in seen_datasets:
    _dataset = SplitLTDataset(_dataset_name)
    _baseline_res = TrainResult.from_dict(
        torch.load(str(_dataset.baseline_eval_file_path), map_location=DEFAULT_DEVICE)
    )
    _res_dict = {"Experiment": "Baseline", "Dataset": _dataset_name}
    if not args.show_adjusted_acc:
        if args.acc_k != 1:
            _res_dict.update(_get_topk_acc(_baseline_res, args))
        else:
            for _split, _split_acc in _baseline_res.test_acc__per__split.items():
                _res_dict[_split.title()] = _split_acc * 100
            _res_dict["Overall"] = _baseline_res.test_metrics["accuracy"] * 100
    else:
        _res_dict.update(_get_adjusted_accs(_baseline_res, args))

    table_rows.insert(0, _res_dict)
    pbar.update(1)
pbar.close()

full_table = pd.DataFrame(table_rows)
for _dataset_name in seen_datasets:
    table = full_table[full_table["Dataset"] == _dataset_name]
    _grouped_table = table.groupby(
        ["Experiment", "Dataset"], group_keys=False, sort=False
    )
    mean_table = _grouped_table.mean().reset_index()
    std_table = _grouped_table.std().reset_index()
    _sig_width__per__split = {}
    for _split in ("Few", "Medium", "Many", "Overall"):
        _sig_width__per__split[_split] = 5 if std_table[_split].max() >= 10 else 4

    hdr_str = f"{'Experiment':{max_exp_name_len + 2}}"
    sep_str = f"{'-' * max_exp_name_len}  "
    for _split in ("Few", "Medium", "Many", "Overall"):
        _split_hdr_size = (
            4  # mean value (xx.x)
            + _sig_width__per__split[_split]  # std value (x.xx or xx.xx)
            + 2  # format characters (<mean>^<std>^)
        )
        _split_title = "Med." if _split == "Medium" else _split
        hdr_str += f"{_split_title:>{_split_hdr_size}}  "  # 2 spaces between columns
        sep_str += f"{'-' * _split_hdr_size}  "

    if len(seen_datasets) > 1:
        _dataset_proper_name = SplitLTDataset(_dataset_name).proper_name
        print(f"\n\n{_dataset_proper_name}:")
        print(f"{'=' * (len(_dataset_proper_name) + 1)}\n")

    print(hdr_str)
    print(sep_str)

    for _mrow, _srow in zip(mean_table.iterrows(), std_table.iterrows()):
        _mrow, _srow = _mrow[1], _srow[1]
        _exp = str(_mrow["Experiment"])
        assert _exp == str(_srow["Experiment"])
        row_str = f"{_exp:{max_exp_name_len + 2}}"
        for _split in ("Few", "Medium", "Many", "Overall"):
            _mu, _sig = _mrow[_split], _srow[_split]
            _mu_str = f"{_mu:4.1f}"
            if not isnan(_sig):
                row_str += f"{_mu_str}^{_sig:{_sig_width__per__split[_split]}.2f}^"
            else:
                row_str += " " * (_sig_width__per__split[_split] + 2) + _mu_str
            row_str += "  "
        print(row_str.rstrip())

    if args.print_csv:
        _col_order = ["Experiment", "Few", "Medium", "Many", "Overall"]
        print()
        print(
            mean_table.to_csv(
                columns=_col_order, index=False, float_format=lambda _f: f"{_f:4.1f}"
            )
        )
