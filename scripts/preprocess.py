#
# Preprocess the MeerTime search mode data
# 2020 Fabian Jankowski
#

import glob
import os.path
import shlex
import subprocess32 as subprocess

import numpy as np


# 1) integrate single files into one longer one per observation
dirs = glob.glob(
    '/fred/oz005/search/J1935+2154/2020-05-1*'
)
dirs = sorted(dirs)

print('Number of directories: {0}'.format(len(dirs)))

for dir in dirs:
    print('Processing directory: {0}'.format(dir))

    files = glob.glob(
        os.path.join(dir, '1*', '2*.sf')
    )
    files = sorted(files)

    print('Number of files: {0}'.format(len(files)))

    # process in bunches
    step = 10
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

        command = 'pfitsUtil_searchmode_combineTime -o {outfile} {filestr}'.format(
            outfile=outfile,
            filestr=' '.join(files_to_process)
        )

        args = shlex.split(command)

        subprocess.check_call(args)
