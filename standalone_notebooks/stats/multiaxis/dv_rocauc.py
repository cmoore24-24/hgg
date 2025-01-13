from coffea.nanoevents import NanoEventsFactory, BaseSchema, PFNanoAODSchema
import json
import numpy as np
import awkward as ak
from coffea import processor
import hist
import warnings
import hist.dask as dhist
import dask
import pickle
import os
from ndcctools.taskvine import DaskVine
import time
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics
import copy
from dask.diagnostics import ProgressBar

full_start = time.time()

if __name__ == "__main__":
    n = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    path = "../../../outputs/cr_investigations/multi_var_hists/"
    name = "cambridge02_pt_mass_argmax.pkl"
    file = path + name
    with open(file, 'rb') as f:
        cring = pickle.load(f)

    #2017 integrated luminosity and QCD cross sections
    IL = 44.99
    xs_170to300 = 103700
    xs_300to470 = 6835
    xs_470to600 = 549.5
    xs_600to800 = 156.5
    xs_800to1000 = 26.22
    xs_1000to1400 = 7.475
    xs_1400to1800 = 0.6482
    xs_1800to2400 = 0.08742
    xs_2400to3200 = 0.005237
    xs_3200toInf = 0.0001353

    signal_dict = {}
    signal_dict['Hgg'] = cring[0]['Hgg']['Hgg']
    signal_dict['Hbb'] = cring[0]['Hbb']['Hbb']
    
    hgg_scaled = signal_dict['Hgg']
    hbb_scaled = signal_dict['Hbb']
    
    #signal scale factors
    scalesHJ = ((44.99*(0.471*1000)*0.0817)/(hgg_scaled['entries']))
    scalesHbb = ((44.99*(0.274*1000)*0.581)/(hbb_scaled['entries']))
    
    #do the scaling 
    hgg_entries = list(hgg_scaled.keys())
    for i in range(1, len(hgg_entries)):
        hgg_scaled[hgg_entries[i]].view(flow=True)[:] *= scalesHJ
    
    hbb_entries = list(hbb_scaled.keys())
    for i in range(1, len(hbb_entries)):
        hbb_scaled[hbb_entries[i]].view(flow=True)[:] *= scalesHbb

    #combine the qcds into a dictionary
    qcd_dict = {}
    qcd_dict['q173'] = cring[0]['QCD_Pt_170to300_TuneCP5_13TeV_pythia8']['QCD_Pt_170to300']
    qcd_dict['q347'] = cring[0]['QCD_Pt_300to470_TuneCP5_13TeV_pythia8']['QCD_Pt_300to470']
    qcd_dict['q476'] = cring[0]['QCD_Pt_470to600_TuneCP5_13TeV_pythia8']['QCD_Pt_470to600']
    qcd_dict['q68'] = cring[0]['QCD_Pt_600to800_TuneCP5_13TeV_pythia8']['QCD_Pt_600to800']
    qcd_dict['q810'] = cring[0]['QCD_Pt_800to1000_TuneCP5_13TeV_pythia8']['QCD_Pt_800to1000']
    qcd_dict['q1014'] = cring[0]['QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8']['QCD_Pt_1000to1400']
    qcd_dict['q1418'] = cring[0]['QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8']['QCD_Pt_1400to1800']
    qcd_dict['q1824'] = cring[0]['QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8']['QCD_Pt_1800to2400']
    qcd_dict['q2432'] = cring[0]['QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8']['QCD_Pt_2400to3200']
    qcd_dict['q32inf'] = cring[0]['QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8']['QCD_Pt_3200toInf']
    
    #QCD scale factors dictionary
    qcd_sf_dict = {}
    qcd_sf_dict['scales170to300'] = (((xs_170to300*1000)*IL)/(qcd_dict['q173']['entries']))
    qcd_sf_dict['scales300to470'] = (((xs_300to470*1000)*IL)/(qcd_dict['q347']['entries']))
    qcd_sf_dict['scales470to600'] = (((xs_470to600*1000)*IL)/(qcd_dict['q476']['entries']))
    qcd_sf_dict['scales600to800'] = (((xs_600to800*1000)*IL)/(qcd_dict['q68']['entries']))
    qcd_sf_dict['scales800to1000'] = (((xs_800to1000*1000)*IL)/(qcd_dict['q810']['entries']))
    qcd_sf_dict['scales1000to1400'] = (((xs_1000to1400*1000)*IL)/(qcd_dict['q1014']['entries']))
    qcd_sf_dict['scales1400to1800'] = (((xs_1400to1800*1000)*IL)/(qcd_dict['q1418']['entries']))
    qcd_sf_dict['scales1800to2400'] = (((xs_1800to2400*1000)*IL)/(qcd_dict['q1824']['entries']))
    qcd_sf_dict['scales2400to3200'] = (((xs_2400to3200*1000)*IL)/(qcd_dict['q2432']['entries']))
    qcd_sf_dict['scales3200toInf'] = (((xs_3200toInf*1000)*IL)/(qcd_dict['q32inf']['entries']))
    
    #scale all the qcd values
    entries = list(qcd_dict['q173'].keys())
    for i in range(0, len(qcd_dict)):
        qcd_range = list(qcd_dict.keys())[i]
        qcd_scales = list(qcd_sf_dict.keys())[i]
        for j in range(1, len(entries)):
            qcd_dict[qcd_range][entries[j]].view(flow=True)[:] *= qcd_sf_dict[qcd_scales]
    
    #combine the qcds into individual variable fields
    qcd_vars_scaled = {}
    for i in range(1, len(entries)):
        temp_hist = qcd_dict['q173'][entries[i]]
        for j in range(1, len(qcd_dict)):
            temp_hist += qcd_dict[list(qcd_dict.keys())[j]][entries[i]]
        qcd_vars_scaled[entries[i]] = temp_hist


    @dask.delayed
    def loop(l, m):
        master_dict = {}

        hgg_copy = copy.copy(hgg_scaled)
        hbb_copy = copy.copy(hbb_scaled)

        hgg_copy['Color_Ring'] = hgg_copy['Color_Ring'][l:m,:,:,:]
        hbb_copy['Color_Ring'] = hbb_copy['Color_Ring'][l:m,:,:,:]

        if (hgg_copy['Color_Ring'].sum().value == 0) or (hbb_copy['Color_Ring'].sum().value == 0):
            category = 'mass_window_' + str(l) + '_' + str(m)
            master_dict[category] = {}
            master_dict[category]['Hgg'] = None
            master_dict[category]['Hbb'] = None

        else:
            hgg = hgg_copy
            hbb = hbb_copy
            
            #get the totals for each histogram
            hgg_totals_dict = {}
            for i in range(1, len(hgg_entries)):
                if len(hgg[hgg_entries[i]].axes) == 1:
                    hgg_totals_dict[hgg_entries[i]] = hgg[hgg_entries[i]][0:len(hgg[hgg_entries[i]].view()):sum]
                else:
                    for j in hgg[hgg_entries[i]].axes.name:
                        hgg_totals_dict[j] = hgg[hgg_entries[i]].project(j)[0:len(hgg[hgg_entries[i]].project(j).view()):sum]   
                    
            hbb_totals_dict = {}
            for i in range(1, len(hbb_entries)):
                if len(hbb[hbb_entries[i]].axes) == 1:
                    hbb_totals_dict[hbb_entries[i]] = hbb[hbb_entries[i]][0:len(hbb[hbb_entries[i]].view()):sum]
                else:
                    for j in hbb[hbb_entries[i]].axes.name:
                        hbb_totals_dict[j] = hbb[hbb_entries[i]].project(j)[0:len(hbb[hbb_entries[i]].project(j).view()):sum]
            
            #get the true positive fractions
            hgg_truth_dict = {}
            for i in range(1, len(hgg_entries)):
                if len(hgg[hgg_entries[i]].axes) == 1:
                    temp_list = []
                    for j in range(1, len(hgg[hgg_entries[i]].view())+1):
                        temp_list.append(hgg[hgg_entries[i]][0:j:sum].value/hgg_totals_dict[hgg_entries[i]].value)
                    hgg_truth_dict[hgg_entries[i]] = temp_list
                else:
                    for j in hgg[hgg_entries[i]].axes.name:
                        temp_list = []
                        for k in range(1, len(hgg[hgg_entries[i]].project(j).view())+1):
                            temp_list.append(hgg[hgg_entries[i]].project(j)[0:k:sum].value/hgg_totals_dict[hgg_entries[i]].value)
                        hgg_truth_dict[j] = temp_list
                            
            hbb_truth_dict = {}
            for i in range(1, len(hbb_entries)):
                if len(hbb[hbb_entries[i]].axes) == 1:
                    temp_list = []
                    for j in range(1, len(hbb[hbb_entries[i]].view())+1):
                        temp_list.append(hbb[hbb_entries[i]][0:j:sum].value/hbb_totals_dict[hbb_entries[i]].value)
                    hbb_truth_dict[hbb_entries[i]] = temp_list
                else:
                    for j in hbb[hbb_entries[i]].axes.name:
                        temp_list = []
                        for k in range(1, len(hbb[hbb_entries[i]].project(j).view())+1):
                            temp_list.append(hbb[hbb_entries[i]].project(j)[0:k:sum].value/hbb_totals_dict[hbb_entries[i]].value)
                        hbb_truth_dict[j] = temp_list
            
            qcd_vars_copy = copy.deepcopy(qcd_vars_scaled)
            qcd_vars_copy['Color_Ring'] = qcd_vars_copy['Color_Ring'][l:m,:,:,:]
            qcd_vars = qcd_vars_copy
            
            #totals for each qcd hist
            qcd_totals_dict = {}
            for i in range(1, len(entries)):
                if len(qcd_vars[entries[i]].axes) == 1:
                    qcd_totals_dict[entries[i]] = qcd_vars[entries[i]][0:len(qcd_vars[entries[i]].view()):sum]
                else:
                    for j in qcd_vars[entries[i]].axes.name:
                        qcd_totals_dict[j] = qcd_vars[entries[i]].project(j)[0:len(qcd_vars[entries[i]].project(j).view()):sum]
            
            #false positive fractions for each qcd variable
            qcd_false_positive_dict = {}
            for i in range(1, len(entries)):
                if len(qcd_vars[entries[i]].axes) == 1:
                    temp_list = []
                    for j in range(1, len(qcd_vars[entries[i]].view())+1):
                        temp_list.append(qcd_vars[entries[i]][0:j:sum].value/qcd_totals_dict[entries[i]].value)
                    qcd_false_positive_dict[entries[i]] = temp_list
                else:
                    for j in qcd_vars[entries[i]].axes.name:
                        temp_list = []
                        for k in range(1, len(qcd_vars[entries[i]].project(j).view())+1):
                            temp_list.append(qcd_vars[entries[i]].project(j)[0:k:sum].value/qcd_totals_dict[entries[i]].value)
                        qcd_false_positive_dict[j] = temp_list
            
            hgg_auc_dict = {}
            hgg_keys = list(hgg_truth_dict.keys())
            for i in range(0, len(hgg_keys)):
                hgg_auc_dict[hgg_keys[i]] = metrics.auc(
                                                    hgg_truth_dict[hgg_keys[i]],
                                                    qcd_false_positive_dict[hgg_keys[i]]
                                                )
            # for i in range(0, len(hgg_keys)):
            #     if hgg_auc_dict[hgg_keys[i]] >= 0.5:
            #         hgg_auc_dict[hgg_keys[i]] = 1 - hgg_auc_dict[hgg_keys[i]]
            
            hbb_auc_dict = {}
            hbb_keys = list(hbb_truth_dict.keys())
            for i in range(0, len(hbb_keys)):
                hbb_auc_dict[hbb_keys[i]] = metrics.auc(
                                                    hbb_truth_dict[hbb_keys[i]],
                                                    qcd_false_positive_dict[hbb_keys[i]]
                                                )
            # for i in range(0, len(hbb_keys)):
            #     if hbb_auc_dict[hbb_keys[i]] >= 0.5:
            #         hbb_auc_dict[hbb_keys[i]] = 1 - hbb_auc_dict[hbb_keys[i]]
            
            master_dict['Hgg'] = hgg_auc_dict
            master_dict['Hbb'] = hbb_auc_dict
    
            return master_dict

    auc_dict = {}

    for l in range(0, 39):
        for m in range(l+2, 43):
            entry_name = 'cr_window_' + str(l) + '_' + str(m)
            auc_dict[entry_name] = loop(l,m)

    print(len(auc_dict))
    computed = dask.compute(
        auc_dict,
        scheduler=n.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )
    output = './outputs/ca/cr_window_' + name
    with open(output, 'wb') as f:
        pickle.dump(computed[0], f)

    
    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))

