#!/bin/bash
WORKDIR=`pwd`

unset PYTHONPATH

mkdir -p weaver
cp -r /scratch365/cmoore24/lund/weaver.tar.gz ./
tar -xzvf weaver.tar.gz -C weaver/
rm weaver.tar.gz
source weaver/bin/activate
export TERMINFO=/usr/share/terminfo
export PATH=$WORKDIR/weaver/bin:$PATH
cp -r /scratch365/cmoore24/weaver_changed ./weaver-core
cd weaver-core
python -m pip install -e .
cd ../

# weaver --predict --data-test 'sig:/scratch365/cmoore24/weaver-core/my_attempt/samples/475to950/hgg/hgg/*' --data-config /scratch365/cmoore24/weaver-core/my_attempt/control.yaml --network-config /scratch365/cmoore24/weaver-core/my_attempt/pn_test.py --model-prefix /scratch365/cmoore24/weaver-core/my_attempt/output/Hgg_train_best_epoch_state.pt --num-workers 6 --gpus '0,1,2,3' --batch-size 1024 --predict-output '/scratch365/cmoore24/weaver-core/my_attempt/ecf_comp/pnet.root' --log /scratch365/cmoore24/weaver-core/my_attempt/predict_output/predict.log #> /scratch365/cmoore24/weaver-core/my_attempt/predict_output/qcd_predict.txt 2>&1

weaver --predict --data-test 'all_classes:/scratch365/cmoore24/weaver-core/my_attempt/samples/475to950/all_classes/hgg_qcd/*' --data-config /scratch365/cmoore24/weaver-core/my_attempt/control.yaml --network-config /scratch365/cmoore24/weaver-core/my_attempt/pn_test.py --model-prefix /scratch365/cmoore24/weaver-core/my_attempt/output/Hgg_train_best_epoch_state.pt --num-workers 6 --gpus '0,1,2,3' --batch-size 1024 --predict-output '/scratch365/cmoore24/weaver-core/my_attempt/predict_output/output.root' --log /scratch365/cmoore24/weaver-core/my_attempt/predict_output/predict.log #> /scratch365/cmoore24/weaver-core/my_attempt/predict_output/qcd_predict.txt 2>&1

#'wqq:/scratch365/cmoore24/weaver-core/my_attempt/samples/wqq_totrain/*'
