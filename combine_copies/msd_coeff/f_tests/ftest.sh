msd=$1

cd msd$1_vs_msd$((${msd}+1))
ln -s ../../../../copy.sh .
./copy.sh

ln -s ../../../../plot-ftest.py .
python3 plot-ftest.py

cd ..
