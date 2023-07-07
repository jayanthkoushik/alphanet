#!/usr/bin/env python3

from argparse import ArgumentParser, SUPPRESS

from corgy import CorgyHelpFormatter

import alphanet.plot

parser = ArgumentParser(formatter_class=CorgyHelpFormatter, usage=SUPPRESS)
sub_parsers = parser.add_subparsers(dest="cmd", required=True)
added_cmds = set()
for cmd_cls in alphanet.plot.BasePlotCmd.__subclasses__():
    if cmd_cls.__name__ in added_cmds:
        continue
    cmd_cls = getattr(alphanet.plot, cmd_cls.__name__)
    _sub_parser = sub_parsers.add_parser(
        cmd_cls.__name__, formatter_class=CorgyHelpFormatter, usage=SUPPRESS
    )
    _sub_parser.set_defaults(corgy=cmd_cls)
    cmd_cls.add_args_to_parser(_sub_parser)
    added_cmds.add(cmd_cls.__name__)

args = parser.parse_args()
cmd = args.corgy.from_dict(vars(args))
cmd()
