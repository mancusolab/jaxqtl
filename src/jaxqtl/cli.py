import argparse
import logging
import os
import sys
from importlib import metadata


def _main(argsv):
    # setup main parser
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # subp = argp.add_subparsers(help="Subcommands: TBD")

    # parse arguments
    args = argp.parse_args(argsv)

    # cmd_str = get_command_string(argsv)

    version = metadata.version("sushie")

    masthead = "===================================" + os.linesep
    masthead += f"             JaxQTL v{version}             " + os.linesep
    masthead += "===================================" + os.linesep

    # setup logging
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    log = logging.getLogger()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)
    log.propagate = False

    # write to stdout unless quiet is set
    if not args.quiet:
        sys.stdout.write(masthead)
        # sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    # disk_log_stream.write(cmd_str)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.logger.addHandler(disk_handler)

    # launch finemap
    args.func(args)

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
