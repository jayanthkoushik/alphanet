#!/usr/bin/env python3

from argparse import SUPPRESS

from alphanet.train import TrainCmd

train_cmd = TrainCmd.parse_from_cmdline(usage=SUPPRESS)
train_cmd()
