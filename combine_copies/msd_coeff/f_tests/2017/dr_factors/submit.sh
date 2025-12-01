#cp ../submit.py .
#cp ../run-ftest.sh .
#cp ../submit.templ.condor .
#cp ../compare.py .

in_arg=0

python submit.py --msd=$in_arg --dr=0 --njobs=10
python submit.py --msd=$in_arg --dr=1 --njobs=10
python submit.py --msd=$in_arg --dr=2 --njobs=10
python submit.py --msd=$in_arg --dr=3 --njobs=10
python submit.py --msd=$in_arg --dr=4 --njobs=10
#python submit.py --msd=5 --njobs=10
