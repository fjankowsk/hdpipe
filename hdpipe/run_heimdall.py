# -*- coding: utf-8 -*-
#
#   Run heimdall single-pulse detection pipeline.
#   2018 - 2020 Fabian Jankowski
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

from hdpipe.general_helpers import (signal_handler, setup_logging)
from hdpipe.version import __version__


def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run heimdall on filterbank files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Filterbank files to process."
    )

    parser.add_argument(
        "-g", "--gpu_id",
        dest="gpu_id",
        type=int,
        choices=[0, 1],
        default=0,
        help="ID of GPU to use."
    )

    parser.add_argument(
        "-z", "--zap_mode",
        dest="zap_mode",
        type=str,
        choices=[
            "None",
            "Lovell_20cm",
            "MeerKAT_20cm"
            ],
        default="None",
        help="Frequency zap mask mode to use."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__
    )

    args = parser.parse_args()

    return args


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

    elif zap_mode == "MeerKAT_20cm":
        # MeerKAT telescope 20cm data
        zap_str = "-zap_chans 0 45 -zap_chans 76 78 -zap_chans 81 83 -zap_chans 93 94 -zap_chans 101 113 -zap_chans 127 226 -zap_chans 263 264 -zap_chans 348 350 -zap_chans 360 361 -zap_chans 369 384 -zap_chans 393 398 -zap_chans 482 483 -zap_chans 488 695 -zap_chans 697 709 -zap_chans 712 720 -zap_chans 722 768 -zap_chans 772 780 -zap_chans 782 785 -zap_chans 787 793 -zap_chans 801 804 -zap_chans 807 825 -zap_chans 836 853 -zap_chans 883 885 -zap_chans 892 942 -zap_chans 949 951 -zap_chans 958 976 -zap_chans 1023 1024"

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

    command = "heimdall -dm 0 5000 -dm_tol 1.05 -output_dir {0} {1} -gpu_id {2} -f {3}".format(
        tempdir,
        zap_str,
        gpu_id,
        filename
    )

    log.info("Heimdall command: {0}".format(command))

    args = shlex.split(command)
    subprocess.check_call(args)

    candfiles = glob.glob(os.path.join(tempdir, "*.cand"))
    total = ""

    for item in candfiles:
        with open(item, "r") as f:
            total += f.read()

    outfile = "{0}.cand".format(
        os.path.splitext(filename)[0]
        )

    with open(outfile, "w") as f:
        f.write(total)

    # clean up
    for item in candfiles:
        os.remove(item)

    os.rmdir(tempdir)


#
# MAIN
#

def main():
    # start signal handler
    signal.signal(signal.SIGINT, signal_handler)

    setup_logging()
    log = logging.getLogger('hdpipe.run_heimdall')

    args = parse_args()

    # sanity check
    for item in args.files:
        if not os.path.isfile(item):
            log.error("The file does not exist: {0}".format(item))
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
            log.error("Heimdall failed on file: {0}, {1}".format(
                item,
                str(e)
                )
            )
        else:
            i += 1

    print("Successfully processed files: {0} ({1})".format(i, len(files)))

    print("All done.")


if __name__ == "__main__":
    main()
