#!/usr/bin/env python3

from argparse import ArgumentParser, SUPPRESS

from corgy import CorgyHelpFormatter

from alphanet.plot import BasePlotCmd

parser = ArgumentParser(formatter_class=CorgyHelpFormatter, usage=SUPPRESS)
sub_parsers = parser.add_subparsers(dest="cmd", required=True)
for cmd_cls in BasePlotCmd.__subclasses__():
    _sub_parser = sub_parsers.add_parser(
        cmd_cls.__name__, formatter_class=CorgyHelpFormatter, usage=SUPPRESS
    )
    _sub_parser.set_defaults(corgy=cmd_cls)
    cmd_cls.add_args_to_parser(_sub_parser)

args = parser.parse_args()
cmd = args.corgy.from_dict(vars(args))
cmd()
