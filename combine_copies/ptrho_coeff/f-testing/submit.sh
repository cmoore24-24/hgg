cp ../submit.py .
cp ../run-ftest.sh .
cp ../submit.templ.condor .
cp ../compare.py .


python submit.py --pt=0 --rho=0 --njobs=10
python submit.py --pt=0 --rho=1 --njobs=10
python submit.py --pt=1 --rho=0 --njobs=10
python submit.py --pt=1 --rho=1 --njobs=10
python submit.py --pt=0 --rho=2 --njobs=10
python submit.py --pt=2 --rho=0 --njobs=10
python submit.py --pt=2 --rho=1 --njobs=10
python submit.py --pt=1 --rho=2 --njobs=10
python submit.py --pt=2 --rho=2 --njobs=10
