import os
import numpy as np
import json

if __name__ == '__main__':

    year = "2016"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    elif "2017" in thisdir:
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    for msd in range(0,6):
        for data_deg in range(0,6):

            print("msd = "+str(msd)+"")
            print("data_deg = "+str(data_deg)+"")
    
            # Make the directory and go there
            thedir = "msd"+str(msd)+str(data_deg)
            if not os.path.isdir(thedir):
                os.mkdir(thedir)
            os.chdir(thedir)
    
    
            os.system("ln -s ../../../../sample_hist.pkl .")
            os.system("ln -s ../../../../min_ral.py .")
            os.system("ln -s ../../../../util.py .")
            os.system("ln -s ../../../../config_Hxx.py .")
    
            # Create your json files of initial values
            if not os.path.isfile("initial_vals_ggf.json"):
    
                initial_vals = (np.full((msd+1),1)).tolist()
                thedict = {}
                thedict["initial_vals"] = initial_vals
    
                with open("initial_vals_ggf.json", "w") as outfile:
                    json.dump(thedict,outfile)
    
            # Create the workspace
            os.system(f"python3 min_ral.py -t sample_hist.pkl -o output --data --degsMC {msd} --degs {data_deg} --basis 'Bernstein,Bernstein' ") 
    
            # Make the workspace
            os.chdir("output/")
            os.system("chmod +rwx build.sh")
            os.system("./build.sh")
            os.chdir("../")
    
            # Run the first fit                                                                                                        
            combine_cmd = "combine -M MultiDimFit -m 125 -d output/model_combined.root --saveWorkspace \
            --setParameters r=0 --freezeParameters r -n \"Snapshot\" \
            --robustFit=1 --cminDefaultMinimizerStrategy 0"
            os.system(combine_cmd)
    
            # Run the goodness of fit
            combine_cmd = "combine -M GoodnessOfFit -m 125 -d higgsCombineSnapshot.MultiDimFit.mH125.root \
            --snapshotName MultiDimFit --bypassFrequentistFit \
            --setParameters r=0 --freezeParameters r \
            -n \"Observed\" --algo \"saturated\" --cminDefaultMinimizerStrategy 0"
            os.system(combine_cmd)

            combine_cmd = "combine -M FitDiagnostics -d output/model_combined.root -m 125 --saveShapes --skipSB"
            os.system(combine_cmd)
    
            os.chdir("../")
