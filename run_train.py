#!/usr/bin/env python3

from alphanet.train import TrainCmd

train_cmd = TrainCmd.parse_from_cmdline()
train_cmd()
