#!/bin/sh
WORKDIR=`pwd`

unset PYTHONPATH

mkdir -p coffea
cp /users/cmoore24/Public/hgg/factory/coffea.tar.gz ./
tar -xzf coffea.tar.gz -C coffea/
rm coffea.tar.gz
source coffea/bin/activate
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/coffea/bin:$PATH

PROC_ID=$1

python -u condor_bootstrap.py --index $PROC_ID