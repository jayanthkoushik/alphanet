import pickle
from abc import ABC, abstractmethod
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import tomli
import torch
from corgy import Corgy
from corgy.types import InputBinFile, InputDirectory, SubClass
from torch import Tensor
from tqdm import trange


class _ClassifierLoader(ABC):
    @abstractmethod
    def load(self, data_file: InputBinFile, device: torch.device) -> torch.nn.Linear:
        raise NotImplementedError


ClassifierLoaderType = SubClass[_ClassifierLoader]
ClassifierLoaderType.allow_base = False


class LinearClassifierLoader(_ClassifierLoader):
    def load(self, data_file, device=torch.device("cpu")):
        data = torch.load(data_file, map_location=device)
        _clf_w__mat = data["weight"]
        _clf_b__vec = data["bias"]

        if _clf_b__vec is None:
            _clf_b__vec = torch.zeros(_clf_w__mat.shape[0], device=device)
        elif _clf_b__vec.shape[0] != _clf_w__mat.shape[0]:
            raise ValueError("weight, bias output shape mismatch")

        clf = torch.nn.Linear(
            in_features=_clf_w__mat.shape[1],
            out_features=_clf_w__mat.shape[0],
            bias=True,
        )
        assert _clf_w__mat.shape == clf.weight.data.shape
        assert _clf_b__vec.shape == clf.bias.data.shape
        clf.weight.data = _clf_w__mat
        clf.bias.data = _clf_b__vec
        return clf


class CombinedClassifierLoader(_ClassifierLoader):
    class CombinedClassifier(torch.nn.Module):
        def __init__(self, classifiers):
            super().__init__()
            self.classifiers = torch.nn.ModuleList(classifiers)

        def forward(self, x):
            _xs = x.unflatten(1, (len(self.classifiers), -1))
            _outs = [
                _classifier_i(_xs[:, _i, :])
                for _i, _classifier_i in enumerate(self.classifiers)
            ]
            return torch.stack(_outs, dim=1).mean(dim=1)

    def load(self, data_file, device=torch.device("cpu")):
        data = torch.load(data_file, map_location=device)
        data_path = Path(data_file.name)

        _classifiers, _classifier_ws, _classifier_bs = [], [], []
        for _data_i in data:
            _loader_i = ClassifierLoaderType(_data_i["classifier_loader"])()
            _classifier_i_path = Path(_data_i["classifier_file"])
            # If the path is not absolute, make it relative to this classifier's path.
            if not _classifier_i_path.is_absolute():
                _classifier_i_path = (data_path.parent / _classifier_i_path).resolve()

            _classifier_i = _loader_i.load(_classifier_i_path, device)
            _classifiers.append(_classifier_i)
            _classifier_ws.append(_classifier_i.weight.data)
            _classifier_bs.append(_classifier_i.bias.data)
            assert _classifier_i.weight.data.shape == _classifier_ws[0].shape

        _n_out_features = _classifier_ws[0].shape[0]
        _n_in_features_ii = _classifier_ws[0].shape[1]
        _n_clfs = len(_classifier_ws)
        _n_in_features = _n_in_features_ii * _n_clfs

        _combined_w__mat = torch.stack(_classifier_ws, dim=1)
        assert _combined_w__mat.shape == (_n_out_features, _n_clfs, _n_in_features_ii)
        _combined_w__mat = _combined_w__mat.flatten(1, 2) / _n_clfs
        assert _combined_w__mat.shape == (_n_out_features, _n_in_features)

        _combined_b__vec = torch.stack(_classifier_bs, dim=1)
        assert _combined_b__vec.shape == (_n_out_features, _n_clfs)
        _combined_b__vec = _combined_b__vec.mean(dim=1)
        assert _combined_b__vec.shape == (_n_out_features,)

        clf = torch.nn.Linear(
            in_features=_n_in_features, out_features=_n_out_features, bias=True
        )
        assert _combined_w__mat.shape == clf.weight.data.shape
        assert _combined_b__vec.shape == clf.bias.data.shape
        clf.weight.data = _combined_w__mat
        clf.bias.data = _combined_b__vec

        # Test the classifier.
        _x = torch.rand(2, _n_in_features, device=device)
        _xs = _x.unflatten(1, (_n_clfs, -1))
        _outs = [
            _classifier_i(_xs[:, _i, :])
            for _i, _classifier_i in enumerate(_classifiers)
        ]
        _yhat_gold = torch.stack(_outs, dim=1).mean(dim=1)
        _yhat = clf(_x)
        assert _yhat.shape == _yhat_gold.shape
        assert torch.allclose(_yhat, _yhat_gold, rtol=1e-3, atol=1e-5)

        return clf


class SplitLTDataset(str):
    feature_files_dir: InputDirectory
    split_files_dir: InputDirectory
    classifier_file: InputBinFile
    classifier_loader: _ClassifierLoader
    baseline_eval_file_path: Path
    label_names_file_path: Optional[Path] = None

    try:
        _configs = tomli.load(InputBinFile("config/datasets.toml"))
    except Exception as e:
        raise RuntimeError(f"failed to load datasets config: {e}") from None

    __choices__ = tuple(_configs.keys())
    __metavar__ = "str"

    def __init__(self, config: str):
        super().__init__()
        if config not in self.__choices__:
            raise ValueError(f"'{config}': no such dataset")
        try:
            self.feature_files_dir = InputDirectory(
                self._configs[config]["feature_files_dir"]
            )
            self.split_files_dir = InputDirectory(
                self._configs[config]["split_files_dir"]
            )
            self.classifier_file = InputBinFile(
                self._configs[config]["classifier_file"]
            )
            self.classifier_loader = ClassifierLoaderType(
                self._configs[config]["classifier_loader"]
            )()
            self.baseline_eval_file_path = self._configs[config]["baseline_eval_file"]
            self.label_names_file_path = self._configs[config].get(
                "label_names_file_path", None
            )
        except KeyError as e:
            raise ValueError(f"config '{config}' missing value for {e}") from None
        except Exception as e:
            raise ValueError(f"invalid config '{config}': {e}") from None

    def load_data(self, datagrp: Literal["train", "val", "test"]):
        return SplitLTDataGroup(self, datagrp)

    def load_classifier(self, device=torch.device("cpu")) -> torch.nn.Linear:
        return self.classifier_loader.load(self.classifier_file, device)


class SplitLTDataGroupInfo(Corgy):
    dataset_name: str
    datagrp: str
    n_features: int
    n_imgs: int
    n_classes: int
    n_base_classes: int
    n_few_classes: int
    n_imgs__per__class: Dict[int, int]
    class__set__per__split: Dict[str, Set[int]]


class SplitLTDataGroup:
    feat__mat: torch.Tensor
    label__seq: List[int]
    label_name__seq: Optional[List[str]] = None
    info: SplitLTDataGroupInfo

    def __init__(
        self, dataset: SplitLTDataset, datagrp: Literal["train", "val", "test"]
    ):
        data_file_names = list(glob(str(dataset.feature_files_dir / f"{datagrp}*.pkl")))
        pbar = trange(
            len(data_file_names),
            desc=f"Loading '{datagrp}' features for '{str(dataset)}'",
            unit="file",
            disable=len(data_file_names) == 1,
        )
        feat__mat_parts = []
        self.label__seq = []
        for _data_file_name in data_file_names:
            with open(_data_file_name, "rb") as _f:
                _data = pickle.load(_f)

            _feats = _data["feats"]
            feat__mat_parts.append(
                _feats if isinstance(_feats, Tensor) else torch.from_numpy(_feats)
            )
            self.label__seq.extend([int(_l) for _l in _data["labels"]])
            pbar.update(1)
        pbar.close()

        self.feat__mat = torch.vstack(feat__mat_parts)

        if dataset.label_names_file_path is not None:
            with open(dataset.label_names_file_path, "r", encoding="utf-8") as _f:
                self.label_name__seq = [_n for _line in _f if (_n := _line.strip())]

        self.info = SplitLTDataGroupInfo()
        self.info.dataset_name = str(dataset)
        self.info.datagrp = datagrp
        self.info.n_imgs__per__class = dict(Counter(self.label__seq))
        self.info.n_features = self.feat__mat.shape[1]
        self.info.n_imgs = self.feat__mat.shape[0]
        self.info.n_classes = len(self.info.n_imgs__per__class)

        self.info.class__set__per__split = {}
        for _split in ("many", "medium", "few"):
            with open(
                dataset.split_files_dir / f"{_split}.txt", "r", encoding="utf-8"
            ) as _f:
                self.info.class__set__per__split[_split] = set(
                    int(_line.strip()) for _line in _f
                )
        self.info.n_few_classes = len(self.info.class__set__per__split["few"])
        self.info.n_base_classes = self.info.n_classes - self.info.n_few_classes

        assert len(self.feat__mat.shape) == 2
        assert self.feat__mat.shape[0] == len(self.label__seq)
        if self.label_name__seq is not None:
            assert len(self.label_name__seq) == self.info.n_classes
        assert sum(self.info.n_imgs__per__class.values()) == self.info.n_imgs
        assert set.union(*self.info.class__set__per__split.values()) == set(
            self.info.n_imgs__per__class.keys()
        )
