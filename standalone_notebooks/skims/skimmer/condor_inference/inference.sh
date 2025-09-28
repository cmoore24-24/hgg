#!/bin/sh
WORKDIR=`pwd`

unset PYTHONPATH

source /users/cmoore24/.bashrc
conda activate coffea
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/coffea/bin:$PATH

WORK_DIR="/project01/ndcms/cmoore24/skims/analysis_skims/2017/nolepton/mc"

cmd="python -u bdt_inference.py --path $WORK_DIR --section $1"

$cmd 2>&1
