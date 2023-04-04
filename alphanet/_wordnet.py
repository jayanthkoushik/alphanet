import logging
import math
from collections import defaultdict
from typing import Dict, List, Literal

import nltk
from nltk.corpus import wordnet
from tqdm import tqdm

nltk.download("wordnet", download_dir=".nltk")
nltk.data.path.append(".nltk")


def get_wordnet_nns_per_class(level: int, for_split: Literal["few", "base", "all"]):
    label_name__per__class = {}
    with open("data/ImageNetLT/label_names.txt", encoding="utf-8") as _f:
        for _i, _line in enumerate(_f):
            label_name__per__class[_i] = _line.strip()
    assert len(label_name__per__class) == 1000

    with open("data/ImageNetLT/splits/few.txt", encoding="utf-8") as _f:
        few_split_class__set = {int(_line.strip()) for _line in _f}
    logging.info("few split classes: %d", len(few_split_class__set))

    synset__per__class = {}
    with open("data/ImageNetLT/labels_full.txt", encoding="utf-8") as _f:
        for _i, _line in enumerate(_f):
            _lid, _label_full = _line.strip().split(" ", maxsplit=1)
            assert _label_full.split(",")[0] == label_name__per__class[_i]
            _synset = wordnet.synset_from_pos_and_offset(_lid[0], int(_lid[1:]))
            synset__per__class[_i] = _synset
    assert len(synset__per__class) == 1000

    nn__seq__per__class: Dict[int, List[int]] = defaultdict(list)
    for _class, _synset in tqdm(
        synset__per__class.items(), desc="Computing WordNet NNs"
    ):
        if for_split != "all":
            if for_split == "few" and _class not in few_split_class__set:
                continue
            if for_split == "base" and _class in few_split_class__set:
                continue

        _hypernym_paths = [list(reversed(_p)) for _p in _synset.hypernym_paths()]
        for _oclass, _osynset in synset__per__class.items():
            if _class == _oclass:
                continue
            _lchs = _synset.lowest_common_hypernyms(_osynset)
            _min_idx = float("inf")
            for _lch in _lchs:
                for _hypernym_path in _hypernym_paths:
                    try:
                        _idx = _hypernym_path.index(_lch)
                    except ValueError:
                        continue
                    if _idx < _min_idx:
                        _min_idx = _idx
            assert not math.isinf(_min_idx)
            if _min_idx <= level:
                nn__seq__per__class[_class].append(_oclass)

    return nn__seq__per__class
