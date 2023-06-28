import logging
from collections import defaultdict
from typing import Dict, List, Optional

import nltk
from corgy.types import OutputTextFile
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset  # pylint: disable=no-name-in-module
from tqdm import tqdm
from typing_extensions import Literal

nltk.download("wordnet", download_dir=".nltk")
nltk.data.path.append(".nltk")


def get_wordnet_nns_per_imgnet_class(
    level: int,
    for_split: Literal["few", "base", "all"],
    few_split_classes_path: str,
    label_names_path: str,
    label_ids_to_names_path: str,
    save_nns_file: Optional[OutputTextFile] = None,
) -> Dict[int, List[int]]:
    label_name__per__class = {}
    with open(label_names_path, encoding="utf-8") as _f:
        for _i, _line in enumerate(_f):
            label_name__per__class[_i] = _line.strip()
    assert len(label_name__per__class) == 1000

    with open(few_split_classes_path, encoding="utf-8") as _f:
        few_split_class__set = {int(_line.strip()) for _line in _f}
    logging.info("few split classes: %d", len(few_split_class__set))

    synset__per__class = {}
    with open(label_ids_to_names_path, encoding="utf-8") as _f:
        for _i, _line in enumerate(_f):
            _lid, _label_full = _line.strip().split(" ", maxsplit=1)
            assert _label_full.split(",")[0] == label_name__per__class[_i]
            _synset = wordnet.synset_from_pos_and_offset(_lid[0], int(_lid[1:]))
            synset__per__class[_i] = _synset
    assert len(synset__per__class) == 1000

    if for_split == "all":
        for_class__set = set(synset__per__class.keys())
    elif for_split == "few":
        for_class__set = few_split_class__set
    else:
        for_class__set = set(synset__per__class.keys()) - few_split_class__set

    nn__seq__per__class: Dict[int, List[int]] = defaultdict(list)
    _nn_dist__seq__per__class: Dict[int, List[int]] = defaultdict(list)
    _nn_common_hyp__seq__per__class: Dict[int, List[Synset]] = defaultdict(list)
    for _class in tqdm(for_class__set, desc="Computing WordNet NNs"):
        _synset = synset__per__class[_class]
        for _oclass, _osynset in synset__per__class.items():
            if _class == _oclass:
                continue
            _dist = _synset.shortest_path_distance(_osynset)
            if _dist <= level:
                nn__seq__per__class[_class].append(_oclass)
                if save_nns_file is not None:
                    _nn_dist__seq__per__class[_class].append(_dist)
                    _nn_common_hyp__seq__per__class[_class].append(
                        _synset.lowest_common_hypernyms(_osynset)[0]
                    )

    if save_nns_file is not None:
        for _class, _nn__seq in nn__seq__per__class.items():
            _label = label_name__per__class[_class]
            print(_label + ":", file=save_nns_file)

            _nn_label__seq = [
                label_name__per__class[_nn_class] for _nn_class in _nn__seq
            ]
            _nn_dist__seq = _nn_dist__seq__per__class[_class]
            _nn_common_hyp__seq = _nn_common_hyp__seq__per__class[_class]
            for _nn_label, _nn_dist, _nn_common_hyp in zip(
                _nn_label__seq, _nn_dist__seq, _nn_common_hyp__seq
            ):
                print(
                    f"\t{_nn_label.lower()} (dist. {_nn_dist!r} via "
                    f"{_nn_common_hyp.name().split('.')[0].lower()!r})",
                    file=save_nns_file,
                )

    return nn__seq__per__class
