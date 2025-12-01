cp /users/cmoore24/Public/combine/test_rhalphalib/tarred/cmssw.tar.gz ./
tar -xvzf cmssw.tar.gz
rm cmssw.tar.gz
cd CMSSW_14_1_0_pre4/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
scram b ProjectRename # this handles linking the already compiled code - do NOT recompile                         
eval `scram runtime -sh` # cmsenv is an alias not on the workers     
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ${_CONDOR_SCRATCH_DIR}
pwd
clear
cp /users/cmoore24/Public/combine/test_rhalphalib/CMSSW_14_1_0_pre4/src/HiggsAnalysis/rhalphalib/run_scripts/ptrho_coeff/f-testing/2017/pt0rho0/ -r ./
clear
ls
cp /users/cmoore24/Public/combine/test_rhalphalib/CMSSW_14_1_0_pre4/src/HiggsAnalysis/rhalphalib/run_scripts/ptrho_coeff/f-testing/2017/compare.py ./
ls
clear
python compare.py --pt=0 --rho=0 --ntoys=50
python3 compare.py --pt=0 --rho=0 --ntoys=50
cd CMSSW_14_1_0_pre4/src/
cmsenv
cd .././
cd ../
clear
python compare.py --pt=0 --rho=0 --ntoys=50
python3 compare.py --pt=0 --rho=0 --ntoys=50
clear
cmssw-el8
exit
