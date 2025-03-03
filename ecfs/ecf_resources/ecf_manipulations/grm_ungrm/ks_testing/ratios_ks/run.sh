#!/bin/sh
WORKDIR=`pwd`

unset PYTHONPATH

mkdir -p weaver
cp -r /scratch365/cmoore24/lund/weaver.tar.gz ./
tar -xzvf weaver.tar.gz -C weaver/
rm weaver.tar.gz
source weaver/bin/activate
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/weaver/bin:$PATH

mkdir -p outputs
export MPLCONFIGDIR=$PWD
echo Starting...

PROC_ID=$1
END=$((500 * (PROC_ID + 1)))

python calculate_ratios.py --proc_id $PROC_ID --end $END