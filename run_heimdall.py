#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import signal
import argparse
import logging
import sys
import os.path
import shlex
import subprocess
from time import sleep

# version info
__version__ = "$Revision$"


def run_heimdall(filename, gpu_id):
    """
    Run heimdall on a filterbank file.
    """

    if not os.path.isfile(filename):
        raise RuntimeError("The file does not exist: {0}".format(filename))

    zap_str = "-zap_chans 48 53 -zap_chans 191 193 -zap_chans 211 230 -zap_chans 252 257 -zap_chans 284 340 -zap_chans 361 365 -zap_chans 409 410 -zap_chans 416 420 -zap_chans 447 451 -zap_chans 461 468 -zap_chans 472 476 -zap_chans 480 484 -zap_chans 668 671 -zap_chans 672 683 -zap_chans 720 725 -zap_chans 731 734"

    command = "heimdall -dm 0 2000 -dm_tol 1.05 {0} -gpu_id {1} -f {2}".format(zap_str, gpu_id, filename)

    logging.info("Heimdall command: {0}".format(command))

    args = shlex.split(command)
    subprocess.check_call(args)


def signal_handler(signum, frame):
    """
    Handle UNIX signals sent to the programme.
    """

    # treat SIGINT/INT/CRTL-C
    if signum == signal.SIGINT:
        logging.warn("SIGINT received, stopping the program.")
        sys.exit(1)

#
# MAIN
#

def main():
    # start signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # handle command line arguments
    parser = argparse.ArgumentParser(description="Run heimdall on filterbank files.")
    parser.add_argument("files", type=str, nargs="+",
    help="Filterbank files to process.")
    parser.add_argument("-g", "--gpu_id", dest="gpu_id", type=int,
    choices=[0, 1], default=0,
    help="Id of GPU to use.")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    # sanity check
    for item in args.files:
        if not os.path.isfile(item):
            logging.error("The file does not exist: {0}".format(item))
            sys.exit(1)

    files = np.sort(args.files)

    print("Number of files to process: {0}".format(len(files)))
    print("Using GPU: {0}".format(args.gpu_id))
    sleep(3)

    for item in files:
        print("Processing: {0}".format(item))

        try:
            run_heimdall(item, args.gpu_id)
        except Exception as e:
            logging.warn("Heimdall failed on file: {0}, {1}".format(item,
            str(e)))

    print("All done.")


# if run directly
if __name__ == "__main__":
    main()
