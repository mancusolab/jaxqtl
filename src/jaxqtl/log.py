# pattern: Imperative Shell

import logging

from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike


_log = logging.getLogger()

_LOG_FORMAT = "[%(asctime)s - %(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CLI_HANDLER_OWNER_ATTRIBUTE = "_jaxqtl_cli_owned"


def _formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


@contextmanager
def cli_logging(
    name: str,
    *,
    path: str | PathLike[str] | None,
    verbose: bool,
) -> Iterator[logging.Logger]:
    r"""Install invocation-owned CLI handlers and remove them on exit.

    **Arguments:**

    name
        Logger name used by the CLI entrypoint.
    path
        Established command output prefix for a ``.log`` sidecar, or ``None``
        for console-only commands such as ``state-factor``.
    verbose
        Use debug logging when true and informational logging otherwise.

    **Returns:**

    A context yielding the configured logger. Every invocation receives fresh
    console and optional disk handlers. Only handlers created by this context
    are removed and closed when the invocation ends.
    """
    logger = logging.getLogger(name)
    previous_level = logger.level
    previous_propagate = logger.propagate
    formatter = _formatter()
    owned_handlers: list[logging.Handler] = []

    console = logging.StreamHandler()
    setattr(console, _CLI_HANDLER_OWNER_ATTRIBUTE, True)
    console.setFormatter(formatter)
    owned_handlers.append(console)
    logger.addHandler(console)

    if path is not None:
        disk = logging.FileHandler(f"{path}.log", mode="w", encoding="utf-8")
        setattr(disk, _CLI_HANDLER_OWNER_ATTRIBUTE, True)
        disk.setFormatter(formatter)
        owned_handlers.append(disk)
        logger.addHandler(disk)

    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    try:
        yield logger
    finally:
        for handler in owned_handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def get_log():
    """get logger for jaxqtl progress"""
    global _log
    logger = _log
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = False
        console = logging.StreamHandler()
        logger.addHandler(console)

        formatter = _formatter()
        console.setFormatter(formatter)

    return logger


def get_logger(name, path=None):
    """get logger for factorgo progress"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = False
        console = logging.StreamHandler()
        logger.addHandler(console)

        # if need millisecond use : %(asctime)s.%(msecs)03d
        formatter = _formatter()
        console.setFormatter(formatter)

        if path is not None:
            disk_log_stream = open(f"{path}.log", "w")
            disk_handler = logging.StreamHandler(disk_log_stream)
            logger.addHandler(disk_handler)
            disk_handler.setFormatter(formatter)

    return logger
