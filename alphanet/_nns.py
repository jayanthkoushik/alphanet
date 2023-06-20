from heapq import nsmallest
from typing import List, Literal, Tuple

import torch
from torch import cdist, Tensor
from torch.nn import functional as F


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
        dist_fn: distance function to use.
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
        norm_for__mat = F.normalize(for__mat, p=2, dim=1)
        norm_neighbors__mat = F.normalize(neighbors__mat, p=2, dim=1)
        cdist__mat = 1 - (norm_for__mat @ norm_neighbors__mat.t())
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
