# -*- coding: utf-8 -*-
#
#   General helper functions.
#   2019 Fabian Jankowski
#

import logging
import signal
import sys


def setup_logging():
    """
    Setup the logging configuration.
    """

    log = logging.getLogger("hdpipe")

    log.setLevel(logging.DEBUG)
    log.propagate = False

    # log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = (
        "%(asctime)s, %(processName)s, %(name)s, %(module)s, %(levelname)s: %(message)s"
    )
    console_formatter = logging.Formatter(fmt)
    console.setFormatter(console_formatter)
    log.addHandler(console)


def signal_handler(signum, frame):
    """
    Handle UNIX signals sent to the program.
    """

    log = logging.getLogger("hdpipe")

    # treat SIGINT/INT/CRTL-C
    if signum == signal.SIGINT:
        log.warn("SIGINT received, stopping the program.")
        sys.exit(1)
