cleart
clear
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
clear
weaver --data-train 'sig:/scratch365/cmoore24/weaver-core/my_attempt/samples/hgg/flat400_totrain/*' 'qcd:/scratch365/cmoore24/weaver-core/my_attempt/samples/qcd/qcd_totrain/*' --data-config /scratch365/cmoore24/weaver-core/my_attempt/control.yaml --network-config /scratch365/cmoore24/weaver-core/my_attempt/pn_test.py --model-prefix '/scratch365/cmoore24/weaver-core/my_attempt/output/Hgg_train' --data-fraction 0.2 --file-fraction 1.0 --num-workers 6 --gpus '0,1,2,3' --batch-size 1024 --start-lr 5e-3 --num-epochs 35 --optimizer adam --log /scratch365/cmoore24/weaver-core/my_attempt/output/train.log
clear
weaver --data-train 'sig:/scratch365/cmoore24/weaver-core/my_attempt/samples/hgg/flat400_totrain/*' 'qcd:/scratch365/cmoore24/weaver-core/my_attempt/samples/qcd/qcd_totrain/*' --data-config /scratch365/cmoore24/weaver-core/my_attempt/control.yaml --network-config /scratch365/cmoore24/weaver-core/my_attempt/pn_test.py --model-prefix '/scratch365/cmoore24/weaver-core/my_attempt/output/Hgg_train' --data-fraction 0.2 --file-fraction 1.0 --num-workers 6 --gpus '0,1,2,3' --batch-size 1024 --start-lr 5e-3 --num-epochs 35 --optimizer adam --log /scratch365/cmoore24/weaver-core/my_attempt/output/train.log
clear
exit
