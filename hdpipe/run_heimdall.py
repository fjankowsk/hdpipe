#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   Run heimdall single-pulse detection pipeline.
#   2018-2019 Fabian Jankowski
#

import argparse
import glob
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from time import sleep

import numpy as np

from hdpipe.version import __version__


def get_zap_str(zap_mode):
    """
    Get the frequency zap mask string for heimdall to use.

    Parameters
    ----------
    zap_mode : str
        Name of the frequency zap mask.

    Returns
    -------
    zap_str : str
        Frequency zap mask string for heimdall.
    """

    if zap_mode == "None":
        # no frequency zapping
        zap_str = ""

    elif zap_mode == "Lovell_20cm":
        # Lovell telescope 20cm data
        zap_str = "-zap_chans 48 53 -zap_chans 191 193 -zap_chans 211 230 -zap_chans 252 257 -zap_chans 284 340 -zap_chans 361 365 -zap_chans 409 410 -zap_chans 416 420 -zap_chans 447 451 -zap_chans 461 468 -zap_chans 472 476 -zap_chans 480 484 -zap_chans 668 671 -zap_chans 672 683 -zap_chans 720 725 -zap_chans 731 734"
    else:
        RuntimeError("Zap mode does not exist: {0}".format(zap_mode))

    return zap_str


def run_heimdall(filename, gpu_id, zap_mode):
    """
    Run heimdall on a filterbank file.

    Parameters
    ----------
    filename : str
        Filenames of filterbank files to process.
    gpu_id : int
        ID of GPU to use.
    zap_mode : str
        Frequency zap mask to use.
    """

    log = logging.getLogger('hdpipe.run_heimdall')

    if not os.path.isfile(filename):
        raise RuntimeError("The file does not exist: {0}".format(filename))

    # get the frequency zap mask string
    zap_str = get_zap_str(zap_mode)

    tempdir = tempfile.mkdtemp()
    log.info("Temp dir: {0}".format(tempdir))

    command = "heimdall -dm 0 5000 -dm_tol 1.05 -output_dir {0} {1} -gpu_id {2} -f {3}".format(tempdir, zap_str, gpu_id, filename)

    log.info("Heimdall command: {0}".format(command))

    args = shlex.split(command)
    subprocess.check_call(args)

    candfiles = glob.glob(os.path.join(tempdir, "*.cand"))
    total = ""

    for item in candfiles:
        with open(item, "r") as f:
            total += f.read()

    outfile = "{0}.cand".format(filename[0:-4])
    with open(outfile, "w") as f:
        f.write(total)

    # clean up
    for item in candfiles:
        os.remove(item)

    os.rmdir(tempdir)


def signal_handler(signum, frame):
    """
    Handle UNIX signals sent to the programme.
    """

    # treat SIGINT/INT/CRTL-C
    if signum == signal.SIGINT:
        logging.warn("SIGINT received, stopping the program.")
        sys.exit(1)


def setup_logging():
    """
    Setup the logging configuration.
    """

    log = logging.getLogger('hdpipe')

    log.setLevel(logging.DEBUG)
    log.propagate = False

    # log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = "%(asctime)s, %(processName)s, %(name)s, %(module)s, %(levelname)s: %(message)s"
    console_formatter = logging.Formatter(fmt)
    console.setFormatter(console_formatter)
    log.addHandler(console)


#
# MAIN
#

def main():
    # start signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # handle command line arguments
    parser = argparse.ArgumentParser(description="Run heimdall on filterbank files.")
    parser.add_argument("files", type=str, nargs="+",
                        help="Filterbank files to process.")
    parser.add_argument("-g", "--gpu_id", dest="gpu_id", type=int,
                        choices=[0, 1], default=0,
                        help="ID of GPU to use.")
    parser.add_argument("-z", "--zap_mode", dest="zap_mode", type=str,
                        choices=["None", "Lovell_20cm"], default="None",
                        help="Frequency zap mask mode to use (default: None).")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger('hdpipe.run_heimdall')

    # sanity check
    for item in args.files:
        if not os.path.isfile(item):
            logging.error("The file does not exist: {0}".format(item))
            sys.exit(1)

    files = np.sort(args.files)

    print("Number of files to process: {0}".format(len(files)))
    print("Using GPU: {0}".format(args.gpu_id))
    print("Zap mode: {0}".format(args.zap_mode))
    sleep(3)

    i = 0

    for item in files:
        print("Processing: {0}".format(item))

        try:
            run_heimdall(item, args.gpu_id, args.zap_mode)
        except Exception as e:
            log.warn("Heimdall failed on file: {0}, {1}".format(item,
            str(e)))
        else:
            i += 1

    print("Successfully processed files: {0} ({1})".format(i, len(files)))

    print("All done.")


# if run directly
if __name__ == "__main__":
    main()
