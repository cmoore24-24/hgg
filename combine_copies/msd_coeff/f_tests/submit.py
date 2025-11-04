#!/usr/bin/python
import os, sys
import subprocess
import argparse

# Main method                                                                          
def main():

    year = "2016"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    elif "2017" in thisdir:
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    cat = "ggf-mc"
    if "vbfhi-mc" in thisdir:
        cat = "vbfhi-mc"
    elif "vbflo-mc" in thisdir:
        cat = "vbflo-mc"
    elif "vbf-mc" in thisdir:
        cat = "vbf-mc"

    parser = argparse.ArgumentParser(description='F-test batch submit')
    parser.add_argument('-m','--msd',nargs='+',help='msd of baseline')
    parser.add_argument('-n','--njobs',nargs='+',help='number of 100 toy jobs to submit')
    args = parser.parse_args()

    msd = int(args.msd[0])
    njobs = int(args.njobs[0])

    loc_base = os.environ['PWD']
    logdir = 'logs'

    tag = "msd" + str(msd)
    script = 'run-ftest.sh'

    homedir = '/users/cmoore24/Public/combine/test_rhalphalib/CMSSW_14_1_0_pre4/src/HiggsAnalysis/rhalphalib/run_scripts/msd_coeff/f_tests/'+year+'/comparisons/'
    outdir = homedir + tag 

    # make local directory
    locdir = logdir
    os.system('mkdir -p  %s' %locdir)

    print('CONDOR work dir: ' + homedir)

    for i in range(0,njobs):
        prefix = tag+"_"+str(i)
        print('Submitting '+prefix)

        condor_templ_file = open("submit.templ.condor")

        transferfiles = "compare.py,msd" + str(msd) + ",msd" + str(msd+1)

        submitargs = str(msd) + " " + outdir + " " + str(i)
    
        localcondor = locdir+'/'+prefix+".condor"
        condor_file = open(localcondor,"w")
        for line in condor_templ_file:
            line=line.replace('TRANSFERFILES',transferfiles)
            line=line.replace('PREFIX',prefix)
            line=line.replace('SUBMITARGS',submitargs)
            condor_file.write(line)
        condor_file.close()
    
        if (os.path.exists('%s.log'  % localcondor)):
            os.system('rm %s.log' % localcondor)
        os.system('condor_submit %s' % localcondor)

        condor_templ_file.close()
    
    return 

if __name__ == "__main__":
    main()
