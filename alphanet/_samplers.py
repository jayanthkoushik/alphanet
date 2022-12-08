import random
from collections import Counter, defaultdict
from functools import reduce
from itertools import zip_longest
from operator import concat
from typing import Dict, List, Tuple

import torch
from corgy import Corgy
from corgy.types import KeyValuePairs, SubClass
from torch import Tensor
from torch.utils.data import TensorDataset
from typing_extensions import Annotated

from alphanet._dataset import SplitLTDataGroup


class _SamplerMeta(type):
    def __repr__(cls):
        return cls.__name__


class BaseSampler(metaclass=_SamplerMeta):
    """Base class for samplers from `SplitLTDataset`."""

    def __init__(self, data: SplitLTDataGroup, *args, **kwargs):
        self.data = data

        # Collect the data indexes for each class, in each split. So,
        # `_idx__seq__per__class__split["few"][10]` is a list of `j`s such that
        # `label__seq[j]` is 10, and class 10 is in the 'few' split.
        self._idx__seq__per__class__esplit: Dict[str, Dict[int, List[int]]] = {
            # 'esplit' = 'many/medium/few/base' where 'base' is 'many' + 'medium'.
            _esplit: defaultdict(list)
            for _esplit in ("many", "medium", "few", "base")
        }
        for _i, _label in enumerate(data.label__seq):
            for _split in ("many", "medium", "few"):
                if _label in data.info.class__set__per__split[_split]:
                    self._idx__seq__per__class__esplit[_split][_label].append(_i)
                    if _split in ("many", "medium"):
                        self._idx__seq__per__class__esplit["base"][_label].append(_i)

        for _split in ("many", "medium", "few"):
            assert data.info.class__set__per__split[_split] == set(
                self._idx__seq__per__class__esplit[_split].keys()
            )
        assert set(self._idx__seq__per__class__esplit["base"].keys()) == (
            set(self._idx__seq__per__class__esplit["many"].keys())
            | set(self._idx__seq__per__class__esplit["medium"].keys())
        )
        for _esplit in ("many", "medium", "few", "base"):
            for _class, _idx__seq in self._idx__seq__per__class__esplit[
                _esplit
            ].items():
                assert all(data.label__seq[_i] == _class for _i in _idx__seq)
                assert len(_idx__seq) == sum(
                    1 for _label in data.label__seq if _label == _class
                )

    def get_raw(self) -> Tuple[List[Tensor], List[int]]:
        sampled_feat__vec__seq = []
        sampled_label__seq = []

        _sampled_idx__seq = self._get()
        for _idx in _sampled_idx__seq:
            sampled_feat__vec__seq.append(self.data.feat__mat[_idx].unsqueeze(0))
            sampled_label__seq.append(self.data.label__seq[_idx])

        return sampled_feat__vec__seq, sampled_label__seq

    def get_dataset(self) -> TensorDataset:
        _feat__vec__seq, _label__seq = self.get_raw()
        return TensorDataset(
            torch.cat(_feat__vec__seq, dim=0), torch.tensor(_label__seq)
        )

    def _get(self) -> List[int]:
        raise NotImplementedError


class _TemplateMixin:
    """Mixin class for creating different samplers based on a single template."""

    _doc_template: str  # this should be defined by a base class

    def __init_subclass__(cls, **kwargs):
        # Create doc string for `cls` by formatting the template with `kwargs`.
        cls.__doc__ = cls._doc_template.format(**kwargs)
        # Add `kwargs` as attributes of `cls`.
        for _k, _v in kwargs.items():
            setattr(cls, _k, _v)
        super().__init_subclass__()


class _AllSplitSampler(BaseSampler):
    _doc_template = """Sampler that gets all points from the '{esplit}' split."""

    esplit: str

    def _get(self):
        return reduce(concat, self._idx__seq__per__class__esplit[self.esplit].values())


# Stub declarations, to be defined dynamically later.


class AllManySampler:
    ...


class AllMediumSampler:
    ...


class AllFewSampler:
    ...


class AllBaseSampler:
    ...


for _esplit in ("many", "medium", "few", "base"):
    _cls_name = f"All{_esplit.title()}Sampler"
    globals()[_cls_name] = type(
        _cls_name, (_AllSplitSampler, _TemplateMixin), {}, esplit=_esplit
    )


class _UniformSplitClassSampler(BaseSampler):
    _doc_template = """Sampler which uniformly samples '{esplit}' classes.

    Sampling is controlled by the ratio to 'few' split classes. Given a ratio
    'r', and 'n' 'few' split classes, the number of sampled classes is 'n * r'. For each
    sampled class, a corresponding 'few' class is sampled, and the number of images
    obtained from the class is equal to the number in the 'few' class.
    """

    esplit: str
    r: float

    def __init__(self, *args, r=1.0, **kwargs):
        if isinstance(r, str):
            r = float(r)
        self.r = r
        super().__init__(*args, **kwargs)

    def _get(self):
        sampled_idx__seq = []

        _n_classes_to_sample = int(
            len(self.data.info.class__set__per__split["few"]) * self.r
        )
        if self.esplit in ("many", "medium"):
            _sampled_class__seq = random.sample(
                self.data.info.class__set__per__split[self.esplit], _n_classes_to_sample
            )
        else:
            _sampled_class__seq = random.sample(
                self.data.info.class__set__per__split["many"]
                | self.data.info.class__set__per__split["medium"],
                _n_classes_to_sample,
            )

        _n_samples_from_class__seq = random.sample(
            list(map(len, self._idx__seq__per__class__esplit["few"].values())),
            _n_classes_to_sample,
        )
        for _class, _n_samples_from_class in zip(
            _sampled_class__seq, _n_samples_from_class__seq
        ):
            # Sample a random subset of images from the sampled class.
            _sampled_class_idx__seq = random.sample(
                self._idx__seq__per__class__esplit["base"][_class],
                _n_samples_from_class,
            )
            sampled_idx__seq.extend(_sampled_class_idx__seq)

        _n_samples__per__class = Counter(
            [self.data.label__seq[_i] for _i in sampled_idx__seq]
        )
        for _class, _n_samples_from_class in zip(
            _sampled_class__seq, _n_samples_from_class__seq
        ):
            assert _n_samples__per__class[_class] == _n_samples_from_class

        return sampled_idx__seq


class UniformManySampler:
    ...


class UniformMediumSampler:
    ...


class UniformBaseSampler:
    ...


for _esplit in ("many", "medium", "base"):
    _cls_name = f"Uniform{_esplit.title()}ClassSampler"
    globals()[_cls_name] = type(
        _cls_name, (_UniformSplitClassSampler, _TemplateMixin), {}, esplit=_esplit
    )


class _BaseWeightedSplitImageSampler(BaseSampler):
    """Base class for weighted random samplers of images from a split."""

    esplit: str
    r: float

    def __init__(self, *args, r=1.0, **kwargs):
        if isinstance(r, str):
            r = float(r)
        self.r = r
        super().__init__(*args, **kwargs)

    def _get_weights(self, idx__seq):
        raise NotImplementedError

    def _get(self):
        sampled_idx__seq = []

        _n_few_imgs = sum(
            self.data.info.n_imgs__per__class[_class]
            for _class in self.data.info.class__set__per__split["few"]
        )
        _n_imgs_to_sample = int(_n_few_imgs * self.r)

        # Get all the indexes for `self.esplit`.
        _split_idx__seq = reduce(
            concat, self._idx__seq__per__class__esplit[self.esplit].values()
        )
        _weights = self._get_weights(_split_idx__seq)
        sampled_idx__seq = random.choices(
            _split_idx__seq, _weights, k=_n_imgs_to_sample
        )

        assert _n_few_imgs == sum(
            1
            for _label in self.data.label__seq
            if _label in self.data.info.class__set__per__split["few"]
        )
        assert len(_split_idx__seq) == sum(
            1
            for _label in self.data.label__seq
            if _label in self._idx__seq__per__class__esplit[self.esplit]
        )

        return sampled_idx__seq


class _UniformSplitImageSampler(_BaseWeightedSplitImageSampler):
    _doc_template = """Sampler that uniformly samples images from the '{esplit}' split.

    The number of images sampled is equal to the number of 'few' split images
    multiplied by the ratio 'r' (provided as an argument).
    """

    def _get_weights(self, idx__seq):
        return None


class UniformManyImageSampler:
    ...


class UniformMediumImageSampler:
    ...


class UniformBaseImageSampler:
    ...


for _esplit in ("many", "medium", "base"):
    _cls_name = f"Uniform{_esplit.title()}ImageSampler"
    globals()[_cls_name] = type(
        _cls_name, (_UniformSplitImageSampler, _TemplateMixin), {}, esplit=_esplit
    )


class _ClassBalancedSplitSampler(_BaseWeightedSplitImageSampler):
    _doc_template = (
        "Sampler that gets '{esplit}' images with weights inverse of their class size"
        """
    The number of images sampled is equal to the number of 'few' split images
    multiplied by the ratio 'r' (provided as an argument).
    """
    )

    def _get_weights(self, idx__seq):
        return [
            (1 / self.data.info.n_imgs__per__class[self.data.label__seq[_idx]])
            for _idx in idx__seq
        ]


class ClassBalancedManySampler:
    ...


class ClassBalancedMediumSampler:
    ...


class ClassBalancedBaseSampler:
    ...


for _esplit in ("many", "medium", "base"):
    _cls_name = f"ClassBalanced{_esplit.title()}Sampler"
    globals()[_cls_name] = type(
        _cls_name, (_ClassBalancedSplitSampler, _TemplateMixin), {}, esplit=_esplit
    )


class CombinedSampler(BaseSampler):
    """Sampler that combines the output of multiple samplers."""

    def __init__(self, data, sampler_cls__seq, sampler_args__seq, *args, **kwargs):
        if 1 < len(sampler_args__seq) < len(sampler_cls__seq):
            raise ValueError(
                f"need {len(sampler_cls__seq)} sets of arguments for samplers, got "
                f"{len(sampler_args__seq)}"
            )
        self._samplers = [
            _sampler_cls(data, *args, **_sampler_args, **kwargs)
            for _sampler_cls, _sampler_args in zip_longest(
                sampler_cls__seq, sampler_args__seq, fillvalue=sampler_args__seq[-1]
            )
        ]
        super().__init__(data, *args, **kwargs)

    def _get(self):
        return reduce(concat, (_sampler._get() for _sampler in self._samplers), [])


class SamplerBuilder(Corgy):
    class _SamplerClassType(SubClass[BaseSampler]):
        @classmethod
        def _choices(cls):
            return tuple(
                _c
                for _c in super()._choices()
                if not str(_c).startswith("_") and str(_c) != "CombinedSampler"
            )

    sampler_classes: Annotated[Tuple[_SamplerClassType, ...], "sampler classes"]
    sampler_args: Annotated[
        Tuple[KeyValuePairs, ...],
        "sampler arguments, one mapping per class--if only one mapping is provided, "
        "it is used for all samplers",
    ] = (KeyValuePairs(""),)

    def build(self, data, *args, **kwargs):
        return CombinedSampler(
            data, self.sampler_classes, self.sampler_args, *args, **kwargs
        )
