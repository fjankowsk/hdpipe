#!/bin/bash
#
# Convert PSRFITS search-mode data to SIGPROC filterbank files
# 2020 Fabian Jankowski
#

files=( /fred/oz005/search/J1935+2154/2020-05-11*/*/2*.sf )
echo ${files}
#printf '%s\n' "${files[@]}"

for file in ${files[@]}; do
    echo ${file}
    output="./${file##*/}.fil"
    echo ${output}
    ./psrfits2fil.py ${file} -o ${output}
done

