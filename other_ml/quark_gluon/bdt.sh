#!/bin/sh
WORKDIR=`pwd`

unset PYTHONPATH

mkdir -p ecf_env/
tar -xzf /scratch365/cmoore24/training/hgg/batch2025/ecf_env.tar.gz -C ecf_env/
source ecf_env/bin/activate
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/ecf_env/bin:$PATH

pip install pynvml
pip install xgboost

export MPLCONFIGDIR=$PWD
echo Starting...

OUTPUT_DIR="/scratch365/cmoore24/training/hgg/quark_gluon/output/all_fatjet"
cp /scratch365/cmoore24/training/hgg/quark_gluon/score_bdt.py ./

mkdir -p "$OUTPUT_DIR"

cmd="python -u score_bdt.py --num_trees 1000 --max_depth 6  --output_dir $OUTPUT_DIR" # --resume $OUTPUT_DIR/$1/$2/$3"

$cmd 2>&1