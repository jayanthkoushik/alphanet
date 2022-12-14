#!/usr/bin/env python3

from corgy import Corgy
from corgy.types import OutputBinFile

from alphanet._dataset import SplitLTDataset
from alphanet.alphanet import AlphaNet
from alphanet.train import TrainCmd, TrainingConfig


class Args(Corgy):
    dataset: SplitLTDataset


args = Args.parse_from_cmdline()
train_cmd = TrainCmd(
    dataset=args.dataset,
    save_file=OutputBinFile(args.dataset.baseline_eval_file_path),
    alphanet=AlphaNet(dummy_mode=True),
    training=TrainingConfig(train_epochs=0, min_epochs=0),
)
train_cmd()
