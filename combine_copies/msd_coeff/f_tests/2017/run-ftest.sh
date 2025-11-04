#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job                   

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

# My job
echo "Arguments passed to the job: "
echo $1
echo $2
echo $3

out=$2
index=$3

python3 compare.py --msd=$1 --ntoys=100 --index=$3

mkdir -p $out

dirs=`ls | grep msd$1_vs`
for d in $dirs; do
    mkdir -p "$out/$d"
    cp -r "$d"/. "$out/$d"/
done