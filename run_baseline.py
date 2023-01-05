#!/usr/bin/env python3

from corgy import Corgy
from corgy.types import OutputBinFile

from alphanet._dataset import SplitLTDataset
from alphanet.alphanet import AlphaNet
from alphanet.train import TrainCmd, TrainingConfig


class Args(Corgy):
    dataset: SplitLTDataset
    train_datagrp: str = "train"
    val_datagrp: str = "val"
    test_datagrp: str = "test"


args = Args.parse_from_cmdline()
train_cmd = TrainCmd(
    dataset=args.dataset,
    save_file=OutputBinFile(args.dataset.baseline_eval_file_path),
    alphanet=AlphaNet(dummy_mode=True),
    training=TrainingConfig(
        train_epochs=0,
        min_epochs=0,
        train_datagrp=args.train_datagrp,
        val_datagrp=args.val_datagrp,
        test_datagrp=args.test_datagrp,
    ),
)
train_cmd()
