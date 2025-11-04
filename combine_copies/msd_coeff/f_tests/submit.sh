cp ../submit.py .
cp ../run-ftest.sh .
cp ../submit.templ.condor .
cp ../compare.py .


python submit.py --msd=0 --njobs=10
python submit.py --msd=1 --njobs=10
python submit.py --msd=2 --njobs=10
python submit.py --msd=3 --njobs=10
python submit.py --msd=4 --njobs=10
#python submit.py --msd=5 --njobs=10
