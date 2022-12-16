#!/usr/bin/env python3

from argparse import ArgumentParser

from corgy import CorgyHelpFormatter

from alphanet.plot import PlotPerClsAccVsSamples

parser = ArgumentParser(formatter_class=CorgyHelpFormatter)
sub_parsers = parser.add_subparsers(dest="cmd", required=True)
for cmd_cls in [PlotPerClsAccVsSamples]:
    _sub_parser = sub_parsers.add_parser(
        cmd_cls.__name__, formatter_class=CorgyHelpFormatter
    )
    _sub_parser.set_defaults(corgy=cmd_cls)
    cmd_cls.add_args_to_parser(_sub_parser)

args = parser.parse_args()
cmd = args.corgy.from_dict(vars(args))
cmd()
