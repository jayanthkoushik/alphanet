from typing import List, Optional, Tuple

import torch
from corgy import Corgy
from torch import nn, Tensor
from torch.nn import functional as F
from typing_extensions import Annotated


class AlphaNet(Corgy, nn.Module, corgy_make_slots=False, corgy_track_bases=False):
    hdims: Annotated[Tuple[int, ...], "hidden dimension sizes for AlphaNet"] = (
        32,
        32,
        32,
    )
    hact: Annotated[
        str,
        "activation function for hidden layers of AlphaNet "
        "(name of a function in `torch.nn.functional`)",
    ] = "leaky_relu"
    dummy_mode: Annotated[bool, "activate dummy mode (return base model)"] = False
    debug: Annotated[bool, "activate debug mode"] = False

    def __init__(self, **kwargs):
        Corgy.__init__(self, **kwargs)
        nn.Module.__init__(self)

        if not self.hdims:
            raise ValueError("at least one hidden layer needed")
        try:
            self.hidden_activation = getattr(F, self.hact)
        except AttributeError:
            raise ValueError(f"unknown activation function '{self.hact}'") from None

        self.source0__vec__seq = None
        self.usource__mat__seq = None
        self.n_targets = None
        self.n_sources_vecs = None
        self.n_source_feats = None
        self._n_generated_alphas = None

        self.conv_layer = None
        self.linear_layer__seq = None

    def set_sources(self, source__mat__seq: List[Tensor]):
        # Validate input.
        if not source__mat__seq:
            raise ValueError("no source classifiers provided")
        if source__mat__seq[0].shape[0] == 1:
            raise ValueError("at least two sources needed")
        if not (
            _source__mat.shape == source__mat__seq[0].shape
            for _source__mat in source__mat__seq
        ):
            raise ValueError("all source classifier matrices must have the same shape")
        if not len(source__mat__seq[0].shape) == 2:
            raise ValueError("source classifier matrices must be 2D")

        self.n_targets = len(source__mat__seq)
        self.n_sources_vecs = source__mat__seq[0].shape[0]
        self.n_source_feats = source__mat__seq[0].shape[1]
        self.source0__vec__seq = [_source__mat[0] for _source__mat in source__mat__seq]
        self._n_generated_alphas = self.n_sources_vecs - 1

        assert all(
            _source0__vec.shape == (self.n_source_feats,)
            for _source0__vec in self.source0__vec__seq
        )

        # Unsqueeze the inputs to create dummy dimensions for batch and channel.
        self.usource__mat__seq = [
            _source__mat.unsqueeze(0).unsqueeze(0) for _source__mat in source__mat__seq
        ]

        # Since the source classifiers are matrices, instead of flattening them to pass
        # through a linear layer, we can compute convolution with filters the same size
        # as the inputs.
        self.conv_layer = nn.Conv2d(
            in_channels=1,
            out_channels=self.hdims[0],
            kernel_size=(self.n_sources_vecs, self.n_source_feats),
        )

        # Create the linear hidden layers.
        self.linear_layer__seq = nn.ModuleList(
            [
                nn.Linear(self.hdims[_i - 1], self.hdims[_i])
                for _i in range(1, len(self.hdims))
            ]
            + [nn.Linear(self.hdims[-1], self._n_generated_alphas)]
        )

    def get_target_alphas(self, target_idx: int) -> Tensor:
        alpha__vec: Tensor

        try:
            _source__mat = self.usource__mat__seq[target_idx]
        except IndexError:
            raise ValueError("target index out of range") from None

        if self.dummy_mode:
            return torch.tensor(
                [1] + [0] * (self.n_sources_vecs - 1),
                dtype=torch.float32,
                device=self.conv_layer.weight.device,
            )

        alpha__vec = self.conv_layer(_source__mat)
        alpha__vec = alpha__vec.squeeze(-1).squeeze(-1)

        if self.debug:
            # Verify that the convolution matches the squeezed dot product.
            with torch.no_grad():
                _gold__vec = Tensor(
                    [
                        (_source__mat[0, 0] * self.conv_layer.weight[_i, 0]).sum()
                        + (
                            self.conv_layer.bias[_i]
                            if self.conv_layer.bias is not None
                            else 0
                        )
                        for _i in range(self.hdims[0])
                    ]
                ).to(alpha__vec.device)
                assert torch.allclose(alpha__vec, _gold__vec, rtol=1e-3, atol=1e-5)

        for _linear_layer in self.linear_layer__seq:
            alpha__vec = self.hidden_activation(alpha__vec)
            alpha__vec = _linear_layer(alpha__vec)

        if self.debug:
            assert alpha__vec.shape == (1, self._n_generated_alphas)

        alpha__vec = alpha__vec.squeeze(0)
        alpha__vec = alpha__vec / alpha__vec.norm(p=1)  # L1 normalize.

        # Add alpha0.
        alpha__vec = torch.concat((torch.ones(1, device=alpha__vec.device), alpha__vec))

        if self.debug:
            assert alpha__vec.shape == (self.n_sources_vecs,)

        return alpha__vec

    def get_target(self, target_idx: int) -> Tensor:
        target__vec: Tensor

        _alpha__vec = self.get_target_alphas(target_idx)
        target__vec = (
            self.usource__mat__seq[target_idx][0, 0] * _alpha__vec.unsqueeze(1)
        ).sum(0)

        if self.debug:
            assert target__vec.shape == (self.n_source_feats,)
            _gold__vec = sum(
                self.usource__mat__seq[target_idx][0, 0, _i] * _alpha__vec[_i]
                for _i in range(self.n_sources_vecs)
            )
            assert isinstance(_gold__vec, Tensor)
            assert torch.allclose(target__vec, _gold__vec, rtol=1e-3, atol=1e-5)

        return target__vec

    def forward(self) -> Tensor:
        return torch.stack([self.get_target(_i) for _i in range(self.n_targets)])


class AlphaNetClassifier(nn.Module):
    def __init__(
        self,
        alphanet: AlphaNet,
        n_base_classes: int,
        ab_class_ordered_idx__vec: Tensor,
        alphanet_bias__init: Optional[Tensor] = None,
        base_clf_weight_init: Optional[Tensor] = None,
        base_clf_bias_init: Optional[Tensor] = None,
        pred_scale: float = 1,
    ):
        super().__init__()
        self.alphanet = alphanet
        self.ab_class_ordered_idx__vec = ab_class_ordered_idx__vec
        self.pred_scale = pred_scale

        if alphanet_bias__init is None:
            alphanet_bias__init = torch.zeros(alphanet.n_targets)
        self.alphanet_b__vec = nn.Parameter(alphanet_bias__init, requires_grad=False)

        if base_clf_weight_init is None:
            base_clf_weight_init = torch.zeros(n_base_classes, alphanet.n_source_feats)
        if base_clf_bias_init is None:
            base_clf_bias_init = torch.zeros(n_base_classes)
        self.base_clf = nn.Linear(alphanet.n_source_feats, n_base_classes)
        self.base_clf.weight.data = base_clf_weight_init
        self.base_clf.bias.data = base_clf_bias_init

        for _param in self.base_clf.parameters():
            _param.requires_grad = False

    def forward(self, x__batch) -> Tensor:
        _alphanet_w__mat = self.alphanet()
        _alphanet_scores__batch = x__batch @ _alphanet_w__mat.t() + self.alphanet_b__vec
        _base_scores__batch = self.base_clf(x__batch)
        return (
            torch.index_select(
                torch.cat((_alphanet_scores__batch, _base_scores__batch), dim=1),
                1,
                self.ab_class_ordered_idx__vec,
            )
            * self.pred_scale
        )

    def get_trained_templates(self) -> Tensor:
        with torch.no_grad():
            _final_few_template__mat = self.alphanet.forward()
        _final_base_template__mat = self.base_clf.weight.data

        # Combine the templates, and order them using 'ab_class_ordered_idx__vec'.
        final_all_template__mat = torch.index_select(
            torch.cat((_final_few_template__mat, _final_base_template__mat), dim=0),
            0,
            self.ab_class_ordered_idx__vec,
        )

        return final_all_template__mat

    def get_learned_alpha_vecs(self) -> Tensor:
        with torch.no_grad():
            _alpha__vec__seq = []
            for _target in range(self.alphanet.n_targets):
                _alpha__vec__seq.append(self.alphanet.get_target_alphas(_target))
            alpha__mat = torch.stack(_alpha__vec__seq)

        assert alpha__mat.shape == (
            self.alphanet.n_targets,
            self.alphanet.n_sources_vecs,
        )
        return alpha__mat
