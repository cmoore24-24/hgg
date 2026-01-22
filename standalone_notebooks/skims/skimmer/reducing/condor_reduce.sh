#!/bin/sh
WORKDIR=`pwd`

unset PYTHONPATH

source /users/cmoore24/.bashrc
conda activate coffea
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/coffea/bin:$PATH

REDUCE_DIRECTORY='/project01/ndcms/cmoore24/skims/analysis_skims/2016APV/nolepton/mc'

echo $1

cmd="python -u reducer.py --path $REDUCE_DIRECTORY --section $1"

$cmd 2>&1
