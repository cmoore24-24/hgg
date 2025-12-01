import os
import numpy as np
import ROOT
import argparse

def gen_toys(infile, ntoys, seed=123456):
    combine_cmd = "combineTool.py -M GenerateOnly -m 125 -d " + infile + "\
    --snapshotName MultiDimFit --bypassFrequentistFit \
    -n \"Toys\" -t "+str(ntoys)+" --saveToys \
    --seed "+str(seed)
    os.system(combine_cmd)

def GoF(infile, ntoys, seed=123456):

    combine_cmd = "combineTool.py -M GoodnessOfFit -m 125 -d " + infile + "\
    --snapshotName MultiDimFit --bypassFrequentistFit \
    -n \"Toys\" -t " + str(ntoys) + " --algo \"saturated\" --toysFile higgsCombineToys.GenerateOnly.mH125."+str(seed)+".root \
    --seed "+str(seed)
    os.system(combine_cmd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='F-test')
    parser.add_argument('-m','--msd',nargs='+',help='msd of ggf baseline')
    parser.add_argument('-d','--dr',nargs='+',help='data residual order')
    parser.add_argument('-n','--ntoys',nargs='+',help='number of toys')
    parser.add_argument('-i','--index',nargs='+',help='index for random seed')
    args = parser.parse_args()

    msd = int(args.msd[0])
    dr = int(args.dr[0])
    ntoys = int(args.ntoys[0])
    index = int(args.index[0])
    seed = 123456+int(args.index[0])*100+31
    
    baseline = "msd"+str(msd)+str(dr)
    alternatives = ["msd" + str(msd + 1) + str(dr), "msd" + str(msd) + str(dr+1)]
    alternatives = sorted(dict.fromkeys(alternatives), key=lambda s: int(s[3:]))

    alternatives += ["msd"+str(msd+1)]
    
    alternatives = list(dict.fromkeys(alternatives))
    
    # iterate in numeric order of the msd value
    for alt in alternatives:
        msd_alt = int(alt[3:])
        # thedir = f"{baseline}_vs_{alt}_job{index}"
        thedir = f"{baseline}_vs_{alt}"

        os.makedirs(thedir, exist_ok=False)
        os.chdir(thedir)

        os.system(f"cp ../{baseline}/higgsCombineSnapshot.MultiDimFit.mH125.root baseline.root")
        os.system(f"cp ../{alt}/higgsCombineSnapshot.MultiDimFit.mH125.root alternative.root")

        gen_toys("baseline.root", ntoys, seed=seed)

        GoF("baseline.root", ntoys, seed=seed)
        os.system(f"mv higgsCombineToys.GoodnessOfFit.mH125.{seed}.root higgsCombineToys.baseline.GoodnessOfFit.mH125.{seed}.root")

        GoF("alternative.root", ntoys, seed=seed)
        os.system(f"mv higgsCombineToys.GoodnessOfFit.mH125.{seed}.root higgsCombineToys.alternative.GoodnessOfFit.mH125.{seed}.root")

        os.chdir('..')