from collections import defaultdict
from heapq import nsmallest
from typing import Dict, List, Literal, Optional, Tuple

import torch
from corgy import Corgy
from corgy.types import OutputBinFile
from torch import cdist, Tensor
from typing_extensions import Annotated

from alphanet.dataset import SplitLTDataGroupInfo, SplitLTDataset


def get_nearest_neighbors(
    k: int,
    *,
    for__mat: Tensor,
    neighbors__mat: Tensor,
    include_self: bool = False,
    dist_fn: Literal["euclidean", "cosine", "random"] = "euclidean",
    sort_by_dist: bool = False,
) -> Tuple[List[List[int]], List[Tensor]]:
    """Get the nearest neighbors for multiple vectors from a pool of neighbors.

    Note:
        Except for the first argument, all arguments to this function are keyword only.

    Args:
        k: number of nearest neighbors to get for each source vector.
        for__mat: n x d tensor with n d-dimensional source vectors.
        neighbors__mat: m x d tensor with m d-dimensional neighbor vectors.
        include_self: whether to include the source vectors as one of the nearest
            neighbors.
        dist_fn: distance function ('euclidean' or 'cosine') to use.
        sort_by_dist: whether to sort the neighbors by distance (ascending).

    Returns a tuple of lists. The first list, of length n, contains sub-lists each of
    length k, with the indices of the nearest neighbors, i.e., the i-th item in the
    first output list is a list of length k, containing the nearest neighbor indices
    for the i-th source vector. If `include_self` is set, then the first index will
    always be -1. The second list has n tensors, each with shape k x d, where the i-th
    tensor contains the k nearest neighbors of the i-th source vector.
    """
    nn_indices__seq: List[List[int]]
    nn__mat__seq: List[Tensor]

    if include_self:
        k -= 1
    if for__mat.shape[1] != neighbors__mat.shape[1]:
        raise ValueError(
            f"`for__mat` and `neighbors__mat` must have the same number of dimensions: "
            f"{for__mat.shape[1]} != {neighbors__mat.shape[1]}"
        )
    if k < 1:
        raise ValueError("`k` must be at least 1 (2 if `include_self` is `True`")
    if k >= neighbors__mat.shape[0]:
        raise ValueError(
            "`k` must be at least 1 (2 if `include_self` is `True`) less than the "
            "number of neighbors"
        )

    ############################################################

    cdist__mat: Tensor

    if dist_fn == "euclidean":
        cdist__mat = cdist(
            for__mat.unsqueeze(0), neighbors__mat.unsqueeze(0), p=2
        ).squeeze(0)
    elif dist_fn == "cosine":
        cdist__mat = 1 - (for__mat @ neighbors__mat.t())
    elif dist_fn == "random":
        cdist__mat = torch.rand(for__mat.shape[0], neighbors__mat.shape[0])
    else:
        raise ValueError(f"unknown distance function: {dist_fn}")

    assert cdist__mat.shape == (for__mat.shape[0], neighbors__mat.shape[0])

    ############################################################

    if sort_by_dist:
        _, _index__mat = cdist__mat.sort(dim=1)
        nn_indices__seq = [_index__vec[:k].tolist() for _index__vec in _index__mat]
    else:
        # Use `heapq` to find the `k` smallest distances.
        nn_indices__seq = [
            # pylint: disable=cell-var-from-loop
            nsmallest(
                k, range(cdist__mat.shape[1]), key=lambda _j: cdist__mat[_i, _j].item()
            )
            for _i in range(cdist__mat.shape[0])
        ]
    nn__mat__seq = [
        torch.index_select(neighbors__mat, dim=0, index=torch.tensor(_indices))
        for _indices in nn_indices__seq
    ]

    assert len(nn_indices__seq) == for__mat.shape[0]
    assert all(len(_nn_indices) == k for _nn_indices in nn_indices__seq)
    assert len(nn__mat__seq) == for__mat.shape[0]
    assert all(
        _nn__mat.shape == (k, neighbors__mat.shape[1]) for _nn__mat in nn__mat__seq
    )
    assert all(
        torch.allclose(_nn__vec, neighbors__mat[_nn_index])
        for _nn_indices, _nn__mat in zip(nn_indices__seq, nn__mat__seq)
        for _nn_index, _nn__vec in zip(_nn_indices, _nn__mat)
    )

    ############################################################

    # Add the source vectors if `include_self` is `True`.
    if include_self:
        nn_indices__seq = [[-1] + _nn_indices for _nn_indices in nn_indices__seq]
        nn__mat__seq = [
            torch.cat((for__mat[_i : _i + 1, :], nn__mat__seq[_i]), dim=0)
            for _i in range(for__mat.shape[0])
        ]

        assert len(nn_indices__seq) == for__mat.shape[0]
        assert all(len(_nn_indices) == k + 1 for _nn_indices in nn_indices__seq)
        assert len(nn__mat__seq) == for__mat.shape[0]
        assert all(
            _nn__mat.shape == (k + 1, neighbors__mat.shape[1])
            for _nn__mat in nn__mat__seq
        )

    return nn_indices__seq, nn__mat__seq


class NNsResult(Corgy):
    data_info: SplitLTDataGroupInfo
    n_neighbors: int
    nn_dist: str
    nn_class__seq__per__fclass: Dict[int, List[int]]
    nn_mean_feat__mat__per__fclass: Dict[int, Tensor]
    nn_clf__per__fclass: Dict[int, torch.nn.Linear]


class NNsCmd(Corgy):
    save_file: Annotated[
        Optional[OutputBinFile], "pickle file to save nearest neighbor results"
    ]
    dataset: Annotated[
        SplitLTDataset, "name of dataset as defined in 'config/datasets.toml'"
    ]
    n_neighbors: Annotated[int, "number of nearest neighbors to find (excluding self)"]
    nn_dist: Annotated[
        Literal["euclidean", "cosine", "random"],
        "distance metric for nearest neighbors",
    ] = "euclidean"

    def __call__(self) -> NNsResult:
        train_data = self.dataset.load_data("train")
        nns_result = NNsResult()

        nns_result.data_info = train_data.info
        nns_result.n_neighbors = self.n_neighbors
        nns_result.nn_dist = self.nn_dist

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
            self.n_neighbors + 1,
            for__mat=mean_feat__mat__per__msplit["few"],
            neighbors__mat=mean_feat__mat__per__msplit["base"],
            include_self=True,
            dist_fn=self.nn_dist,
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
            assert len(_nn_class__seq) == self.n_neighbors
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
        _full_clf = self.dataset.load_classifier()
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
                in_features=train_data.info.n_features,
                out_features=self.n_neighbors + 1,
            )
            nns_result.nn_clf__per__fclass[_fclass].weight.data = _nn_clf_w__mat
            nns_result.nn_clf__per__fclass[_fclass].bias.data = _nn_clf_b__vec

        if self.save_file is not None:
            torch.save(nns_result.as_dict(recursive=True), self.save_file)
            self.save_file.close()
        return nns_result


if __name__ == "__main__":
    cmd = NNsCmd.parse_from_cmdline()
    cmd()
