WORKDIR=`pwd`
unset PYTHONPATH
source /users/cmoore24/.bashrc
conda activate coffea
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/coffea/bin:$PATH
WORK_DIR="/project01/ndcms/cmoore24/skims/analysis_skims/2017/nolepton/hgg"
clear
python bdt_inference.py --path $WORK_DIR/keep0.parquet 
exit
