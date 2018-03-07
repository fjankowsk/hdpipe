#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import signal
import argparse
import logging
import sys
import os.path

# version info
__version__ = "$Revision$"


def load_data(filename):
    """
    Load and parse heimdall candidate output.
    """

    dtype = [("snr","float"), ("samp_nr","int"), ("time","float"), ("filter","int"),
            ("dmtrial","int"), ("dm","float"),
            ("cluster_nr","int"), ("start","int"), ("end","int")]

    data = np.genfromtxt(filename, dtype=dtype)

    return data


def plot_candidates(t_data, filename):
    """
    Plot heimdall candidate output.
    """

    data = np.copy(t_data)

    # remove all low-snr candidates and the ones that are really wide
    mask = (data["snr"] > 9.0) & (data["filter"] <= 8) & (data["dm"] > 0)
    data = data[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sc = ax.scatter(data["dm"] + 1, data["snr"],
    c=data["filter"],
    marker="o")
    plt.colorbar(sc, label="Filter number")

    ax.set_xscale("log")
    ax.grid()
    ax.set_xlabel("DM+1 [pc/cm3]")
    ax.set_ylabel("S/N")
    ax.set_title("{0}".format(filename))

    fig.tight_layout()

    fig.savefig("{0}.png".format(filename), bbox_inches="tight")

    # close the figure in order not
    # to consume too much memory
    plt.close(fig)


def plot_candidate_timeline(t_data, filename):
    """
    Plot heimdall candidate output as a timeline
    """

    data = np.copy(t_data)

    # remove all low-snr candidates and the ones that are really wide
    #mask = (data["snr"] > 9.0) & (data["filter"] <= 8) & (data["dm"] > 0)
    #data = data[mask]

    data = np.sort(data, order="time")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sc = ax.scatter(data["time"], data["snr"],
    c=data["filter"],
    s=data["dm"]*40.0/2000.0,
    marker="o")
    plt.colorbar(sc, label="Filter number")

    ax.grid()
    ax.set_xlabel("time from start of integration")
    ax.set_ylabel("S/N")
    ax.set_title("{0}".format(filename))

    fig.tight_layout()


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
    parser = argparse.ArgumentParser(description="View heimdall candidates.")
    parser.add_argument("files", type=str, nargs="+",
    help="Candidate files to process.")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    # sanity check
    for item in args.files:
        if not os.path.isfile(item):
            logging.error("The file does not exist: {0}".format(item))
            sys.exit(1)

    for item in args.files:
        print(item)
        data = load_data(item)
        plot_candidates(data, item)
        plot_candidate_timeline(data, item)

    plt.show()

# if run directly
if __name__ == "__main__":
    main()
