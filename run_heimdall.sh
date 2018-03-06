#!/bin/bash
#
# Run heimdall on filterbank files
# 2017 Fabian Jankowski
#

# frequency channels to zap
# for Lovell telescope, 1400 MHz
zap_list="-zap_chans 48 53 -zap_chans 191 193 -zap_chans 211 230 -zap_chans
252 257 -zap_chans 284 340 -zap_chans 361 365 -zap_chans 409 410 -zap_chans
416 420 -zap_chans 447 451 -zap_chans 461 468 -zap_chans 472 476 -zap_chans
480 484 -zap_chans 668 683 -zap_chans 720 725 -zap_chans 731 734"

fils=(*.fil)
echo "Number of files to process: ${#fils[@]}"
sleep 3

for file in ${fils[@]}; do
    echo "Processing: ${file}"
    heimdall -dm 0 2000 -dm_tol 1.05 ${zap_list} \
    -f ${file}
done
