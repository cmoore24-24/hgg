#!/bin/sh

mkdir -p outputs_grm
export MPLCONFIGDIR=$PWD
echo Starting...

WORKDIR=`pwd`

cmd="python3 -u trainer-groom-exp.py --batch_size $1 --starting_nodes $2 --num_layers $3"

$cmd 2>&1
