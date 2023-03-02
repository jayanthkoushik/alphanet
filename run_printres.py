#!/usr/bin/env python3

from argparse import SUPPRESS
from math import isnan
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from corgy import Corgy
from corgy.types import InputDirectory
from tqdm import trange

from alphanet._dataset import SplitLTDataset
from alphanet._utils import DEFAULT_DEVICE
from alphanet.train import TrainResult


class Args(Corgy):
    base_res_dir: InputDirectory
    rel_exp_paths: Tuple[Path]
    exp_names: Optional[Tuple[str]] = None
    res_files_pattern: str = "**/*.pth"
    exp_prefix: str = ""
    exp_suffix: str = ""


args = Args.parse_from_cmdline(usage=SUPPRESS)
table_rows: List[Dict[str, Any]] = []
max_exp_name_len = len("Experiment")
seen_datasets = set()

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
        for _split, _split_acc in _res.test_acc__per__split.items():
            _res_dict[_split.title()] = _split_acc * 100
        _res_dict["Overall"] = _res.test_metrics["accuracy"] * 100
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
    for _split, _split_acc in _baseline_res.test_acc__per__split.items():
        _res_dict[_split.title()] = _split_acc * 100
    _res_dict["Overall"] = _baseline_res.test_metrics["accuracy"] * 100

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
            + 3  # format characters (<mean>^±<std>^)
        )
        _split_title = "Med." if _split == "Medium" else _split
        hdr_str += f"{_split_title:{_split_hdr_size + 2}}"  # 2 spaces between columns
        sep_str += f"{'-' * _split_hdr_size}  "

    print(f"\n\n{_dataset_name}:")
    print(f"{'=' * (len(_dataset_name) + 1)}\n")
    print(hdr_str)
    print(sep_str)

    for _mrow, _srow in zip(mean_table.iterrows(), std_table.iterrows()):
        _mrow, _srow = _mrow[1], _srow[1]
        _exp = str(_mrow["Experiment"])
        assert _exp == str(_srow["Experiment"])
        row_str = f"{_exp:{max_exp_name_len + 2}}"
        for _split in ("Few", "Medium", "Many", "Overall"):
            _mu, _sig = _mrow[_split], _srow[_split]
            row_str += f"{_mu:4.1f}"
            if not isnan(_sig):
                row_str += f"^±{_sig:{_sig_width__per__split[_split]}.2f}^"
            else:
                row_str += " " * (_sig_width__per__split[_split] + 3)
            row_str += "  "
        print(row_str.rstrip())

    _col_order = ["Experiment", "Few", "Medium", "Many", "Overall"]
    print()
    print(
        mean_table.to_csv(
            columns=_col_order, index=False, float_format=lambda _f: f"{_f:4.1f}"
        )
    )
