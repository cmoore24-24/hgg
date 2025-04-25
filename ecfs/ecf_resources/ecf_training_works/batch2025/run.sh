#!/bin/sh
WORKDIR=`pwd`

unset PYTHONPATH

mkdir -p ecf_env/
tar -xzf /scratch365/cmoore24/training/hgg/batch2025/ecf_env.tar.gz -C ecf_env/
source ecf_env/bin/activate
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/weaver/bin:$PATH

pip install pynvml

export MPLCONFIGDIR=$PWD
echo Starting...

OUTPUT_DIR="/scratch365/cmoore24/training/hgg/batch2025/outputs/v4/lr_test"

mkdir -p "$OUTPUT_DIR/$1/$2/$3"

cmd="python -u ecf_trainer_v4.py --batch_size $1 --starting_nodes $2 --num_layers $3 --output_dir $OUTPUT_DIR/$1/$2/$3" # --resume $OUTPUT_DIR/$1/$2/$3"

$cmd 2>&1
