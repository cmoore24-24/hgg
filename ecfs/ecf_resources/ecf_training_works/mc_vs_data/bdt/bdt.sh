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

OUTPUT_DIR="/scratch365/cmoore24/training/hgg/ecf_vs_data/bdt/output/1291"
cp /scratch365/cmoore24/training/hgg/ecf_vs_data/bdt/score_bdt.py ./

mkdir -p "$OUTPUT_DIR/$1/$2/$3"

cmd="python -u score_bdt.py --num_trees $1 --max_depth $2  --output_dir $OUTPUT_DIR/$1/$2/$3" # --resume $OUTPUT_DIR/$1/$2/$3"

$cmd 2>&1