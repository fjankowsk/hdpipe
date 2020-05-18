#
# Preprocess the MeerTime search mode data
# 2020 Fabian Jankowski
#

import glob
import os.path
import shlex
import subprocess32 as subprocess
import sys

import numpy as np

from psrfits2fil import main as presto_convert


def integrate_psrfits():
    """
    Integrate single PSRFITS files into longer ones
    """

    dirs = glob.glob(
        '/fred/oz005/search/J1935+2154/2020-05-1*'
    )
    dirs = sorted(dirs)

    print('Number of directories: {0}'.format(len(dirs)))
    sys.stdout.flush()

    for dir in dirs:
        print('Processing directory: {0}'.format(dir))

        files = glob.glob(
            os.path.join(dir, '1*', '2*.sf')
        )
        files = sorted(files)

        print('Number of files: {0}'.format(len(files)))
        sys.stdout.flush()

        # process in bunches
        step = 30
        nbatch = int(np.ceil(len(files) / float(step)))

        for ibatch in range(nbatch):
            start = ibatch * step
            end = (ibatch + 1) * step

            if end >= len(files):
                end = len(files) - 1

            files_to_process = files[start:end]
            print('Files to process: {0}'.format(len(files_to_process)))

            outfile = os.path.join(
                '.',
                os.path.basename(files_to_process[0])
            )
            outfile = os.path.abspath(outfile)
            print('Output file: {0}'.format(outfile))
            sys.stdout.flush()

            command = 'pfitsUtil_searchmode_combineTime -o {outfile} {filestr}'.format(
                outfile=outfile,
                filestr=' '.join(files_to_process)
            )

            args = shlex.split(command)

            subprocess.check_call(args)
            sys.stdout.flush()


def convert_to_filterbank():
    """
    Convert the psrfits data to filterbank.
    """

    files = glob.glob(
        os.path.join('.', '2*.sf')
    )
    files = sorted(files)

    print('Number of PSRFITS files: {0}'.format(len(files)))

    nbit = 8
    apply_weights = True
    apply_scales = True
    apply_offsets = True

    for item in files:
        infile = os.path.abspath(item)
        outfile = '{0}.fil'.format(infile)

        presto_convert(
            infile,
            outfile,
            nbit,
            apply_weights,
            apply_scales,
            apply_offsets
        )


#
# MAIN
#

def main():
    integrate_psrfits()
    convert_to_filterbank()


if __name__ == "__main__":
    main()
