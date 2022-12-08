import logging
import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import tomli
import torch
from corgy import Corgy
from corgy.types import InputBinFile, SubClass
from torch import Tensor
from tqdm import trange

from alphanet._nns import get_nearest_neighbors


class _ClassifierLoader(ABC):
    @abstractmethod
    def load(self, data_file: Path, device: torch.device) -> torch.nn.Linear:
        raise NotImplementedError


ClassifierLoaderType = SubClass[_ClassifierLoader]
ClassifierLoaderType.allow_base = False


class LinearClassifierLoader(_ClassifierLoader):
    def load(self, data_file, device=torch.device("cpu")):
        data = torch.load(str(data_file), map_location=device)
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
        data = torch.load(str(data_file), map_location=device)
        data_path = Path(data_file.name)

        _classifiers, _classifier_ws, _classifier_bs = [], [], []
        for _data_i in data:
            _loader_i = ClassifierLoaderType(_data_i["classifier_loader"])()
            _classifier_i_path = Path(_data_i["classifier_file"])
            # If the path is not absolute, make it relative to this classifier's path.
            if not _classifier_i_path.is_absolute():
                _classifier_i_path = str(
                    (data_path.parent / _classifier_i_path).resolve()
                )

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
    feature_files_dir: Path
    split_files_dir: Path
    classifier_file: Path
    classifier_loader: _ClassifierLoader
    nn_files_dir: Path
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
            self.feature_files_dir = Path(self._configs[config]["feature_files_dir"])
            self.split_files_dir = Path(self._configs[config]["split_files_dir"])
            self.classifier_file = Path(self._configs[config]["classifier_file"])
            self.classifier_loader = ClassifierLoaderType(
                self._configs[config]["classifier_loader"]
            )()
            self.nn_files_dir = Path(self._configs[config]["nn_files_dir"])
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

    def load_nns(
        self, nn_dist: str, n_neighbors: int, device=torch.device("cpu"), generate=False
    ):
        _nns_file = self.nn_files_dir / f"{nn_dist}_{n_neighbors}.pth"
        if not _nns_file.exists():
            if not generate:
                raise ValueError(
                    "nns file not found; call with `generate=True` to generate results"
                )
            logging.warning("nns file not found: generating results")
            self.nn_files_dir.mkdir(parents=True, exist_ok=True)
            NNsResult._generate(self, nn_dist, n_neighbors)

        return NNsResult.from_dict(torch.load(_nns_file, map_location=device))


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


class NNsResult(Corgy):
    n_neighbors: int
    nn_dist: str
    nn_class__seq__per__fclass: Dict[int, List[int]]
    nn_mean_feat__mat__per__fclass: Dict[int, Tensor]
    nn_clf__per__fclass: Dict[int, torch.nn.Linear]

    @classmethod
    def _generate(
        cls, dataset: SplitLTDataset, nn_dist: str, n_neighbors: int
    ) -> "NNsResult":
        train_data = dataset.load_data("train")
        nns_result = NNsResult()
        nns_result.n_neighbors = n_neighbors
        nns_result.nn_dist = nn_dist

        # Steps:
        # 1. Find mean feature vector for each class.
        # 2. Find nearest neighbors for 'few' split classes.
        # 3. Convert nearest neighbor indices to class labels.
        # 4. Find classifiers associated with nearest neighbor classes.

        ############################################################

        class__seq__per__msplit: Dict[str, List[int]] = {}
        mean_feat__mat__per__msplit: Dict[str, Tensor] = {}

        # Find mean feature vector for each class, separated by split.
        _mean_feat__vec__per__class__msplit: Dict[str, Dict[int, Tensor]] = {
            "base": defaultdict(lambda: torch.zeros(train_data.info.n_features)),
            "few": defaultdict(lambda: torch.zeros(train_data.info.n_features)),
        }
        for _label, _feat__vec in zip(train_data.label__seq, train_data.feat__mat):
            _msplit = (
                "few"
                if _label in train_data.info.class__set__per__split["few"]
                else "base"
            )
            _mean_feat__vec__per__class__msplit[_msplit][_label] += (
                _feat__vec / train_data.info.n_imgs__per__class[_label]
            )

        # Convert mean vector dictionaries to lists of indices, and matrices.
        # This is to get the data as required by `get_nearest_neighbors`.
        # So, `mean_feat__mat__per__msplit['few']` is a matrix where each row is the
        # mean feature vector for the corresponding class label in
        # `class__seq__per__msplit['few']`.
        for _msplit in ("base", "few"):

            _class__seq, _mean_feat__mats = zip(
                *_mean_feat__vec__per__class__msplit[_msplit].items()
            )
            class__seq__per__msplit[_msplit] = list(_class__seq)
            mean_feat__mat__per__msplit[_msplit] = torch.stack(_mean_feat__mats)

        assert all(
            _mean_feat__vec.shape == (train_data.info.n_features,)
            for _msplit in ("base", "few")
            for _mean_feat__vec in (
                _mean_feat__vec__per__class__msplit[_msplit].values()
            )
        )

        ############################################################

        nn_indices__seq: List[List[int]]
        nn_mean_feat__mat__seq: List[Tensor]

        # Compute nearest neighbors. This call will return relative indices for each
        # 'few' class, as well as corresponding mean feature matrices.
        # `nn_indices__seq[i][j]` is the index of the `j`th nearest neighbor for class
        # `class__seq__per__msplit['few'][i]`, and `nn_mean_feat__mat__seq[i][j]` is the
        # mean feature vector of the `j`th nearest neighbor for that class.
        nn_indices__seq, nn_mean_feat__mat__seq = get_nearest_neighbors(
            n_neighbors + 1,
            for__mat=mean_feat__mat__per__msplit["few"],
            neighbors__mat=mean_feat__mat__per__msplit["base"],
            include_self=True,
            dist_fn=nn_dist,  # type: ignore
            sort_by_dist=False,
        )
        # Note: in the above, `j > 0` since the first nearest neighbor is the class
        # itself, so `nn_indices__seq[i][0] == -1` for all `i`.
        assert all(_nn_indices[0] == -1 for _nn_indices in nn_indices__seq)

        ############################################################

        # Map 'few' split labels to nn labels and nn features. Within `nn_data`,
        # `nn_class__seq__per__fclass[fclass][i]` is the class label of the `i+1`th
        # nearest neighbor for `fclass`, and
        # `nn_mean_feat__mat__per__fclass[fclass][i+1]` is the corresponding mean
        # feature vector. `nn_mean_feat__mat__per__fclass[fclass][0]` is the mean
        # feature vector for `fclass` itself.
        nns_result.nn_class__seq__per__fclass = {  # fclass = 'few' split class
            class__seq__per__msplit["few"][_i]: [
                class__seq__per__msplit["base"][_nn_index]
                for _nn_index in nn_indices__seq[_i][1:]
            ]
            for _i in range(len(nn_indices__seq))
        }
        nns_result.nn_mean_feat__mat__per__fclass = dict(
            zip(class__seq__per__msplit["few"], nn_mean_feat__mat__seq)
        )

        assert (
            nns_result.nn_class__seq__per__fclass.keys()
            == train_data.info.class__set__per__split["few"]
        )
        for _nn_class__seq in nns_result.nn_class__seq__per__fclass.values():
            assert len(_nn_class__seq) == n_neighbors
            assert all(
                _nn_class in train_data.info.class__set__per__split["many"]
                or _nn_class in train_data.info.class__set__per__split["medium"]
                for _nn_class in _nn_class__seq
            )
        assert (
            nns_result.nn_mean_feat__mat__per__fclass.keys()
            == train_data.info.class__set__per__split["few"]
        )
        for _fclass in nns_result.nn_class__seq__per__fclass:
            assert (
                len(nns_result.nn_class__seq__per__fclass[_fclass])
                == len(nns_result.nn_mean_feat__mat__per__fclass[_fclass]) - 1
            )
            for _nn_class, _nn_mean_feat__vec in zip(
                nns_result.nn_class__seq__per__fclass[_fclass],
                nns_result.nn_mean_feat__mat__per__fclass[_fclass][1:],
            ):
                assert torch.allclose(
                    _nn_mean_feat__vec,
                    _mean_feat__vec__per__class__msplit["base"][_nn_class],
                )

        ############################################################

        # Map 'few' split labels to nearest neighbor classifiers. Within `nn_data`,
        # `nn_clf__per__fclass[fclass]` is a classifier (W, b) composed of the
        # classifiers of the nearest neighbors of `fclass`.
        _full_clf = dataset.load_classifier()
        nns_result.nn_clf__per__fclass = {}
        for _fclass, _nn_class__seq in nns_result.nn_class__seq__per__fclass.items():
            # Get (W, b) for the nearest neighbors of `fclass`.
            _nn_clf_w__mat = torch.stack(
                [_full_clf.weight[_fclass]]  # W for `fclass` itself
                + [_full_clf.weight[_nn_class] for _nn_class in _nn_class__seq]
            )
            _nn_clf_b__vec = torch.stack(
                [_full_clf.bias[_fclass]]  # b for `fclass` itself
                + [_full_clf.bias[_nn_class] for _nn_class in _nn_class__seq]
            )
            nns_result.nn_clf__per__fclass[_fclass] = torch.nn.Linear(
                in_features=train_data.info.n_features, out_features=n_neighbors + 1
            )
            nns_result.nn_clf__per__fclass[_fclass].weight.data = _nn_clf_w__mat
            nns_result.nn_clf__per__fclass[_fclass].bias.data = _nn_clf_b__vec

        _save_file = dataset.nn_files_dir / f"{nn_dist}_{n_neighbors}.pth"
        torch.save(nns_result.as_dict(recursive=True), str(_save_file))
        return nns_result
