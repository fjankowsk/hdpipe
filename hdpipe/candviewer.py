#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   Visualize heimdall single-pulse candidates.
#   2018-2019 Fabian Jankowski
#

import argparse
import logging
import math
import os.path
import shlex
import signal
import subprocess
import sys
import tempfile
from time import sleep

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from hdpipe.version import __version__


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

    data = np.sort(data, order="total_time")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sc = ax.scatter(data["total_time"], data["dm"]+1,
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


def get_zap_file(zap_mode):
    """
    Get the name of the psrsh zap file to use.

    Parameters
    ----------
    zap_mode : str
        Zap mode to use.

    Returns
    -------
    zap_file : str
        Absolute filename of zap file to use.
    """

    if zap_mode == "None":
        # no zapping
        zap_file = os.path.expanduser("~/freq_zap_masks/zap_none.psh")

    elif zap_mode == "Lovell_20cm":
        # Lovell 20cm data
        # this works for 672 and 800 channel data
        zap_file = os.path.expanduser("~/freq_zap_masks/zap_20cm.psh")

    else:
        RuntimeError("Zap mask mode unknown: {0}".format(zap_mode))

    return zap_file


def plot_candidate_dspsr(fil_file, cand_file, cand_nr, sample, filter, dm, snr,
zap_mode, nchan=0, nbin=0, length=0):
    """
    Plot a candidate using dspsr.
    """

    if not os.path.isfile(fil_file):
        raise RuntimeError("Filterbank file does not exist: {0}".format(fil_file))

    # determine data parameters
    cmd = "header {0} -tsamp -tobs -tstart -nchans -foff -fch1".format(fil_file)
    args = shlex.split(cmd)
    raw = subprocess.check_output(args, encoding="ASCII")
    info = raw.split("\n")

    samp_time = float(info[0].strip())
    samp_time *= 1E-6
    tobs = float(info[1].strip())
    tstart = float(info[2].strip())
    rec_nchan = int(info[3].strip())
    foff = float(info[4].strip())
    fch1 = float(info[5].strip())
    rec_bw = float(abs(foff * rec_nchan))
    rec_cfreq = float(fch1 + 0.5*rec_nchan * foff)

    cand_time = samp_time * sample
    # determine mjd of candidate
    cand_mjd = float(tstart + cand_time/(60*60*24.0))

    logging.info("Data parameters: {0}, {1}, {2}, {3}, {4}, {5} MJD".format(samp_time, tobs,
                                                                        rec_cfreq, rec_nchan,
                                                                        rec_bw, tstart))
    logging.info("Candidate parameters: {0} s, {1} MJD".format(cand_time, cand_mjd))

    # use the absolute path here
    fil_file = os.path.abspath(fil_file)

    cmd = "dmsmear -f {0} -b {1} -n {2} -d {3} -q".format(rec_cfreq,
                                                          rec_bw, rec_nchan, dm)
    args = shlex.split(cmd)
    result = subprocess.check_output(args, stderr=subprocess.STDOUT,
                                     encoding="ASCII")

    cand_band_smear = float(result.strip())
    logging.info("cand_band_smear: {0}".format(cand_band_smear))

    cand_filter_time = (2 ** filter) * samp_time
    logging.info("Filter, cand_filter_time: {0}, {1}".format(filter,
                                                             cand_filter_time))

    cand_smearing = float(cand_band_smear) + float(cand_filter_time)
    cand_start_time = cand_time - (0.5 * cand_smearing)
    cand_tot_time = 2.0 * cand_smearing

    if length != 0:
        cand_tot_time = length

    # determine the bin width, based on heimdalls filter width
    if nbin == 0:
        bin_width = cand_filter_time
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

    logging.info("dspsr cmd: {0}".format(cmd))

    # create a temporary working directory
    workdir = tempfile.mkdtemp()
    logging.info("workdir: {0}".format(workdir))

    args = shlex.split(cmd)
    result = subprocess.check_output(args, stderr=subprocess.STDOUT,
                                     encoding="ASCII", cwd=workdir)

    archive = result.split("seconds: ")[1]
    archive = archive.strip()
    archive = os.path.join(workdir, "{0}.ar".format(archive))
    logging.debug(archive)

    count = 10 
    while ((not os.path.exists(archive)) and count > 0):
        logging.warn("Archive file does not exist: {0}".format(archive))
        sleep(1)
        count = count - 1

    if nchan == 0:
        # determine number of channels based on SNR
        nchan = int(round(math.pow(float(snr)/4.0,2)))
    if nchan < 2:
        nchan = 2

    # round nchan to closest power of 2
    #nchan_base2 = int(round(math.log(nchan, 2)))
    #nchan = math.pow(2,nchan_base2)

    if nchan > 512:
        nchan = 512

    # get the zap file to use
    zap_file = get_zap_file(zap_mode)
    if not os.path.isfile(zap_file):
        raise RuntimeError("The zap file does not exist: {0}".format(zap_file))

    info_str_l = r"Cand {0}\n{1}\n{2:.5f}".format(cand_nr,
    os.path.basename(archive)[0:-3], cand_mjd)
    logging.info(info_str_l)

    info_str_r = r"S/N {0:.1f}; DM {1:.1f}; w {2:.1f} ms\n{3}\n{4}".format(snr,
    dm, cand_filter_time*1E3, os.path.basename(fil_file),
    os.path.basename(cand_file))
    logging.info(info_str_r)

    outfile = os.path.join(".",
    "c{0:0>4}_{1}.png".format(cand_nr, os.path.basename(archive)[0:-3]))

    cmd = "psrplot -p freq+ -J {0} -j 'F {1:.0f}' -c above:l='{2}' -c above:c='' -c above:r='{3}' -c x:unit=ms -c y:reverse=1 -D {4}/PNG {5}".format(zap_file,
    nchan, info_str_l, info_str_r, outfile, archive)

    logging.info("psrplot cmd: {0}".format(cmd))
    args = shlex.split(cmd)
    subprocess.check_call(args)

    # clean up
    if os.path.exists(archive):
        os.remove(archive)
    os.rmdir(workdir)


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
    parser.add_argument("candfiles", type=str, nargs="+",
                        help="Candidate files to process.")
    parser.add_argument("-o", "--output", action="store_true", dest="output",
                        default=False, help="Output plots to file rather than to screen.")
    parser.add_argument("-z", "--zap_mode", dest="zap_mode", type=str,
                        choices=["None", "Lovell_20cm"], default="None",
                        help="Frequency zap mask mode to use (default: None).")
    parser.add_argument("-n", "--nchan", type=int, dest="nchan",
                        default=32, help="Scrunch to this many frequency channels (default: 32).")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    # sanity check
    for item in args.candfiles:
        if not os.path.isfile(item):
            logging.error("The file does not exist: {0}".format(item))
            sys.exit(1)

    if not args.nchan >= 2:
        logging.error("Nchan must be greater than 2: {0}".format(args.nchan))
        sys.exit(1)

    candfiles = np.sort(args.candfiles)

    data = None
    i = 1

    for item in candfiles:
        print("Processing: {0}".format(item))
        part = load_data(item)

        part["cand_file"] = item
        part["fil_file"] = "{0}.fil".format(item[0:-5])
        part["total_time"] =  part["time"] + (i - 1)*60.0

        part = remove_bad_cands(part)

        if data is None:
            data = np.copy(part)
        else:
            data = np.concatenate((data, part))

        i += 1
    
    plot_clusters(data, item, args.output)
    plot_candidates(data, item, args.output)
    plot_candidate_timeline(data, item, args.output)

    # remove all low-snr candidates and the ones that are really wide
    good = remove_bad_cands(data)
    print("Number of good candidates: {0}".format(len(good)))

    # sort candidates by snr for plotting
    good = np.sort(good, order="snr")
    good = np.flipud(good)

    sleep(3)

    print("Plotting good candidates.")

    # sanity check
    for item in good:
        if os.path.isfile(item["cand_file"]) \
        and os.path.isfile(item["fil_file"]):
            pass
        else:
            raise RuntimeError("Cand or fil file do not exist: {0}, {1}".format(item["cand_file"], item["fil_file"]))

    cand_nr = 1

    for item in good:
        try:
            plot_candidate_dspsr(item["fil_file"], item["cand_file"], cand_nr,
                                 item["samp_nr"], item["filter"], item["dm"], item["snr"],
                                 args.zap_mode, nchan=args.nchan)
        except subprocess.CalledProcessError as e:
            print("An error occurred: {0}".format(str(e)))

        cand_nr += 1

    print("All done.")

    if not args.output:
        plt.show()


# if run directly
if __name__ == "__main__":
    main()