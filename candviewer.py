#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import signal
import argparse
import logging
import sys
import os.path
import shlex
import subprocess
import tempfile
from time import sleep
import math

# version info
__version__ = "$Revision$"


def dtype_add_fields(data, dtype):
    """
    Add fields to dtype.
    data is that array where the fields should be added.
    dtype is a dtype definition of the new fields.
    Return is a numpy dtype object.
    """

    dtype = np.dtype(dtype)

    # check that field names are not already there
    for field in dtype.names:
        if field in data.dtype.names:
            logging.error("Field is already there: {0}".format(field))
            sys.exit(1)

    r = list(data.dtype.descr) + list(dtype.descr)

    r = np.dtype(r)

    return r


def load_data(filename):
    """
    Load and parse heimdall candidate output.
    """

    dtype = [("snr","float"), ("samp_nr","int"), ("time","float"),
            ("filter","int"),
            ("dmtrial","int"), ("dm","float"),
            ("n_clusters","int"), ("start","int"), ("end","int")]

    temp = np.genfromtxt(filename, dtype=dtype)
    temp = np.atleast_1d(temp)

    dtype = [("cand_file","|U4096"), ("fil_file","|U4096"),
    ("total_time","float")]
    new_dtype = dtype_add_fields(temp, dtype)

    data = np.zeros(len(temp), dtype=new_dtype)

    # fill in
    for field in data.dtype.names:
        if field in temp.dtype.names:
            data[field] = temp[field]

    data["cand_file"] = filename

    return data


def remove_bad_cands(t_data):
    """
    Remove candidates that are RFI.
    """

    data = np.copy(t_data)

    # remove all low-snr candidates and the ones that are really wide
    mask = (data["snr"] > 9.0) & (data["filter"] <= 8) & \
    (data["dm"] > 100) & (data["n_clusters"] > 4)
    data = data[mask]

    return data


def plot_candidates(t_data, filename, output_plots):
    """
    Plot heimdall candidate output.
    """

    data = np.copy(t_data)

    # remove all low-snr candidates and the ones that are really wide
    data = remove_bad_cands(data)

    print("Number of candidates: {0}".format(len(data)))

    if not len(data) > 0:
        return

    data = np.sort(data, order=["snr", "dm", "filter"])

    if len(data) > 0:
        for item in data:
            print("{0}, {1}, {2}, {3}".format(item["snr"], item["dm"],
            item["filter"], item["cand_file"]))

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

    if output_plots:
        fig.savefig("{0}.png".format(filename), bbox_inches="tight")

        # close the figure in order not
        # to consume too much memory
        plt.close(fig)


def plot_clusters(t_data, filename, output_plots):
    """
    Plot heimdall candidate output.
    """

    data = np.copy(t_data)

    if not len(data) > 0:
        return

    fig = plt.figure()
    ax1 = fig.add_subplot(311)

    ax1.scatter(data["dm"], data["n_clusters"])
    ax1.grid(True)
    ax1.set_yscale("log")

    ax2 = fig.add_subplot(312)
    ax2.scatter(data["snr"], data["n_clusters"])
    ax2.grid(True)
    ax2.set_yscale("log")

    ax3 = fig.add_subplot(313)
    ax3.scatter(data["snr"], data["n_clusters"])
    ax3.grid(True)
    ax3.set_yscale("log")


def plot_candidate_timeline(t_data, filename, output_plots):
    """
    Plot heimdall candidate output as a timeline
    """

    data = np.copy(t_data)

    # remove all low-snr candidates and the ones that are really wide
    data = remove_bad_cands(data)

    if not len(data) > 0:
        return

    data = np.sort(data, order="time")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sc = ax.scatter(data["time"], data["dm"]+1,
    c=2**data["filter"],
    norm=LogNorm(),
    s=60.0*data["snr"]/np.max(data["snr"]),
    marker="o",
    edgecolor="black",
    lw=0.6,
    cmap="Reds")
    plt.colorbar(sc, label="Filter number")

    ax.grid()
    ax.set_xlabel("time [s]")
    ax.set_ylabel("DM+1 [pc/cm3]")
    ax.set_title("{0}".format(filename))
    ax.set_yscale("log")

    fig.tight_layout()


def output_pulse_plot(t_data):
    """
    Run dspsr and psrplot to generate pulse plot.
    """

    data = np.copy(t_data)

    if not len(data) == 1:
        raise RuntimeError("Please provide a single candidate at a time.")

    telescope = "jb"
    start = data["start"]
    end = start + 1.5

    command = "dspsr -S {0} -T {1} -D {2} -O {3} {4}".format(start, end,
    data["dm"], outfile, data["fil_name"])

    # split into correct tokens for Popen 
    args = shlex.split(command)
    logging.debug(args)

    # spawn process
    result = subprocess.check_call(args)


def plot_candidate_dspsr(fil_file, sample, filter, dm, snr, nchan=0, nbin=0,
length=0):
    """
    Plot a candidate using dspsr.
    """

    # for Lovell telescope
    samp_time = 0.000256

    if not os.path.isfile(fil_file):
        raise RuntimeError("Filterbank file does not exist: {0}".format(fil_file))

    # use the absolute path here
    fil_file = os.path.abspath(fil_file)

    cand_time = samp_time * sample

    cmd = "dmsmear -f 1382 -b 400 -n 1024 -d {0} -q".format(dm)
    args = shlex.split(cmd)
    result = subprocess.check_output(args, stderr=subprocess.STDOUT,
    encoding="ASCII")

    cand_band_smear = float(result.strip())
    print("plotCandDspsr: cand_band_smear={0}".format(cand_band_smear))

    cand_filter_time = (2 ** filter) * samp_time

    cand_smearing = float(cand_band_smear) + float(cand_filter_time)
    cand_start_time = cand_time - (0.5 * cand_smearing)
    cand_tot_time = 2.0 * cand_smearing

    if length != 0:
        cand_tot_time = length

    # determine the bin width, based on heimdalls filter width
    if nbin == 0:
        bin_width = samp_time * (2 ** filter)
        nbin = int(cand_tot_time / bin_width)

    if nbin < 16:
        nbin = 16

    if nbin > 1024:
        nbin = 1024

    cmd = "dspsr " + fil_file + " -S " + str(cand_start_time) + \
        " -b " + str(nbin) + \
        " -T " + str(cand_tot_time) + \
        " -c " + str(cand_tot_time) + \
        " -D " + str(dm) + \
        " -U 1" + \
        " -cepoch start" + \
        " -q -Q"

    # create a temporary working directory
    workdir = tempfile.mkdtemp()
    print("plotCandDspsr: workdir={0}".format(workdir))

    print("plotCandDspsr: {0}".format(cmd))
    args = shlex.split(cmd)
    result = subprocess.check_output(args, stderr=subprocess.STDOUT,
    encoding="ASCII", cwd=workdir)

    archive = result.split("seconds: ")[1]
    archive = archive.strip()
    archive = os.path.join(workdir, archive + ".ar")
    print(archive)

    count = 10 
    while ((not os.path.exists(archive)) and count > 0):
        print("plotCandDspsr: archive file does not exist: {0}".format(archive))
        sleep(1)
        count = count - 1

    if nchan == 0:
        # determine number of channels based on SNR
        nchan = int(round(math.pow(float(snr)/4.0,2)))
    if nchan < 2:
        nchan = 2

    nchan_base2 = int(round(math.log(nchan,2)))
    nchan = math.pow(2,nchan_base2)

    if nchan > 512:
        nchan = 512

    title = "DM=" + str(dm) + " Length=" + str(cand_filter_time*1000) + \
            "ms Epoch=" + str(cand_start_time)

    cmd = "psrplot -p freq+ -c above:c='' -c x:unit=ms -j 'F {0:.0f}' -D -/PNG {1}".format(nchan, archive)

    print("plotCandDspsr: {0}".format(cmd))
    args = shlex.split(cmd)
    binary_data = subprocess.check_output(args)

    with open("test.png", "wb") as f:
        f.write(binary_data)
    
    if os.path.exists(archive):
      os.remove(archive)
    os.rmdir(workdir)
    
    return binary_data


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
    parser.add_argument("-o", "--output", action="store_true", dest="output",
    default=False, help="Output plots to file rather than to screen.")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    # sanity check
    for item in args.files:
        if not os.path.isfile(item):
            logging.error("The file does not exist: {0}".format(item))
            sys.exit(1)

    data = None
    i = 0

    for item in args.files:
        print("Processing: {0}".format(item))
        part = load_data(item)

        if data is None:
            data = np.copy(part)
        else:
            part["time"] += i*60.0
            i += 1
            data = np.concatenate((data, part))
    
    plot_clusters(data, item, args.output)
    plot_candidates(data, item, args.output)
    plot_candidate_timeline(data, item, args.output)

    plot_candidate_dspsr("58182_36871_J2124-3358_000000.fil", 62272,
    6, 237, 13.36)

    print("All done.")

    if not args.output:
        plt.show()


# if run directly
if __name__ == "__main__":
    main()
