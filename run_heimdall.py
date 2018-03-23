#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import signal
import argparse
import logging
import sys
import os
import shlex
import subprocess
import tempfile
import glob
from time import sleep

# version info
__version__ = "$Revision$"


def run_heimdall(filename, gpu_id):
    """
    Run heimdall on a filterbank file.
    """

    if not os.path.isfile(filename):
        raise RuntimeError("The file does not exist: {0}".format(filename))

    # Lovell telescope 20cm data
    zap_str = "-zap_chans 191 193 -zap_chans 211 230 -zap_chans 252 257 -zap_chans 284 340 -zap_chans 361 365 -zap_chans 409 451 -zap_chans 461 484 -zap_chans 668 671 -zap_chans 672 683 -zap_chans 720 725 -zap_chans 731 734"

    tempdir = tempfile.mkdtemp()
    logging.info("Temp dir: {0}".format(tempdir))

    command = "heimdall -dm 0 5000 -dm_tol 1.05 -output_dir {0} {1} -gpu_id {2} -f {3}".format(tempdir, zap_str, gpu_id, filename)

    logging.info("Heimdall command: {0}".format(command))

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

    i = 0

    for item in files:
        print("Processing: {0}".format(item))

        try:
            run_heimdall(item, args.gpu_id)
        except Exception as e:
            logging.warn("Heimdall failed on file: {0}, {1}".format(item,
            str(e)))
        else:
            i += 1

    print("Successfully processed files: {0} ({1})".format(i, len(files)))

    print("All done.")


# if run directly
if __name__ == "__main__":
    main()
