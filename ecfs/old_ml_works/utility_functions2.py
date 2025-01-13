import numpy as np
import awkward as ak
import hist
import warnings
import pickle
from coffea.ml_tools.torch_wrapper import torch_wrapper
import matplotlib.pyplot as plt
import hist
from sklearn.metrics import roc_curve, auc
import math
import os
import uproot
import json
import pandas as pd
import pyarrow as pa

def signal_read_in(signal_name, msd_up, msd_down, pt_up, pt_down):
    #signal = ak.firsts(ak.from_parquet(f'/project01/ndcms/cmoore24/skims/full_cut_skims/sole_vars/{signal_name}.parquet'))
    signal = ak.from_parquet(f'/scratch365/cmoore24/training/data/with_300/{signal_name}.parquet')
    signal = signal[signal.msoftdrop < msd_up]
    signal = signal[signal.msoftdrop > msd_down]
    signal = signal[signal.pt < pt_up]
    signal = signal[signal.pt > pt_down]
    signal['mratio'] = signal.mass/signal.msoftdrop
    model1 = f'../models_and_scalers/full_cut/qcd/{signal_name}_traced_model.pt'
    # model2 = f'../models_and_scalers/full_cut/diboson/{signal_name}_traced_model.pt'
    # model3 = f'../models_and_scalers/hgg_vs_hbb/{signal_name}_traced_model.pt'
    # model4 = f'../models_and_scalers/singletop/{signal_name}_traced_model.pt'
    with open(f'../models_and_scalers/full_cut/qcd/{signal_name}_scaler.pkl', 'rb') as f:
        scaler1 = pickle.load(f)
    # with open(f'../models_and_scalers/full_cut/diboson/{signal_name}_scaler.pkl', 'rb') as f:
    #     scaler2 = pickle.load(f)
    # with open(f'../models_and_scalers/hgg_vs_hbb/{signal_name}_scaler.pkl', 'rb') as f:
    #     scaler3 = pickle.load(f)
    # with open(f'../models_and_scalers/singletop/{signal_name}_scaler.pkl', 'rb') as f:
    #     scaler4 = pickle.load(f)
    return signal, model1, scaler1,

def bkg_read_in(bkg_name, msd_up, msd_down, pt_up, pt_down):
    #bkg = ak.firsts(ak.from_parquet(f'/project01/ndcms/cmoore24/skims/full_cut_skims/sole_vars/{bkg_name}.parquet'))
    bkg = ak.from_parquet(f'/scratch365/cmoore24/training/data/with_300/{bkg_name}.parquet')
    bkg = bkg[bkg.msoftdrop < msd_up]
    bkg = bkg[bkg.msoftdrop > msd_down]
    bkg = bkg[bkg.pt < pt_up]
    bkg = bkg[bkg.pt > pt_down]
    bkg['mratio'] = bkg.mass/bkg.msoftdrop
    return bkg

def score_plotter(sample_scores, label, pltrange):
    plt.hist(sample_scores, bins=100, range=pltrange, label=label, density=True, histtype='step')

def make_hist(sample, sample_score):
    min = abs(math.floor(ak.min(sample_score)))
    max = math.ceil(ak.max(sample_score))
    bins = min + max
    score_hist = hist.Hist.new.Reg(bins, -min, max, name=f"NN_Score", label=f"{sample}_NN", overflow=False, underflow=False).Weight()
    score_hist.fill(NN_Score=sample_score)
    return score_hist

def roc_maker(signal_array, bkg_array, sig_score_hist, bkg_score_hist, 
              sig_score, bkg_score, title, pt_up, pt_down, msd_up, msd_down, include_pn=False):
    bkg_zeros = ak.zeros_like(bkg_score)
    sig_ones = ak.ones_like(sig_score)
    combined = ak.concatenate([bkg_score,sig_score])
    combined_truth = ak.concatenate([bkg_zeros, sig_ones])

    bkg_total = bkg_score_hist[0:bkg_score_hist.size:sum]
    sig_total = sig_score_hist[0:sig_score_hist.size:sum]

    wp_dict = {}

    bkg_min = abs(int(bkg_score_hist.to_numpy()[1][0]))
    sig_min = abs(int(sig_score_hist.to_numpy()[1][0]))
    
    for i in range(-10, 10, 5):
        bkg_wp_value = bkg_score_hist[bkg_min+i:bkg_score_hist.size:sum]
        bkg_ratio = bkg_wp_value.value/bkg_total.value
    
        sig_wp_value = sig_score_hist[sig_min+i:sig_score_hist.size:sum]
        sig_ratio = sig_wp_value.value/sig_total.value
        wp_dict[i] = [sig_ratio, bkg_ratio]

    if include_pn:
        signal_array['isSignal'] = np.ones_like(signal_array.particleNetMD_QCD)
        bkg_array['isSignal'] = np.zeros_like(bkg_array.particleNetMD_QCD)
        arg1 = ak.concatenate([signal_array.isSignal, bkg_array.isSignal])
        arg2 = ak.concatenate([signal_array.particleNetMD_QCD, bkg_array.particleNetMD_QCD])  
        arg3 = ak.concatenate([signal_array.particleNet_HbbvsQCD, bkg_array.particleNet_HbbvsQCD])
        
        fpr2, tpr2, thresholds2 = roc_curve(arg1, arg2)
        fpr3, tpr3, thresholds3 = roc_curve(arg1, arg3)
        
        roc_auc2 = auc(tpr2, fpr2)
        roc_auc3 = auc(fpr3, tpr3)

    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(combined_truth, combined)
    roc_auc = auc(fpr, tpr)
    #ax.set_yscale("log")
    ax.plot(tpr, fpr, lw=2, color="cyan", label="auc = %.3f" % (roc_auc))
    if include_pn:
        ax.plot(tpr2, fpr2, lw=2, color="red", label="PN_MD_QCD auc = %.3f" % (1-roc_auc2))
        ax.plot(fpr3, tpr3, lw=2, color="orange", label="PN_HbbvQCD auc = %.3f" % (1-roc_auc3))
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle="--", lw=2, color="k", label="random chance")
    ax.set_xlim([0, 1.0])
    #ax.set_ylim([1e-2, 1.0])
    ax.set_xlabel("true positive rate")
    ax.set_ylabel("false positive rate")
    ax.set_title(f"{title} ROC curve")

    for i in wp_dict:
        ax.plot(wp_dict[i][0], wp_dict[i][1], 'o', label=f'WP = {str(i)}')
    
    # ax.axhline(y=1e-2, color='grey', linestyle='--')
    # ax.axhline(y=1e-3, color='grey', linestyle='--')
    # ax.axhline(y=1e-4, color='grey', linestyle='--')
    # ax.axhline(y=1e-5, color='grey', linestyle='--')
    
    ax.legend(loc="lower right", bbox_to_anchor=(1.5, 0.4))
    
    plt.gcf().text(0.95, 0.3, f'{pt_down} < pt < {pt_up}', fontsize=14)
    plt.gcf().text(0.95, 0.2, f'{msd_down} < msd < {msd_up}', fontsize=14)
    plt.gcf().text(0.95, 0.1, '|η| < 2.4', fontsize=14)
    
    plt.show()

def bkg_output_hist(sample, sample_score1, wp1, sample_score2, wp2, fail):
    if fail:
        mask = (sample_score1 < wp1) & (sample_score2 < wp2)
        sample_cut_msd = bkg_dict[sample]['msoftdrop'][mask]
    else:
        mask = (sample_score1 > wp1) & (sample_score2 > wp2)
        sample_cut_msd = bkg_dict[sample]['msoftdrop'][mask]
    msd_hist = hist.Hist.new.Reg(40, msd_down, msd_up, name=f"msd", label=f"{sample} MSD").Weight()
    msd_hist.fill(msd=sample_cut_msd)
    return msd_hist

def nan_remover(array):
    temp_dict = {}
    for k in array.fields:
        if array[k].fields == []:
            try:
                temp_dict[k] = ak.flatten(array[k])
            except:
                temp_dict[k] = array[k]
        else:
            for m in array[k].fields:
                temp_dict[m] = array[k][m]
    nan_mask = np.ones_like(temp_dict[list(temp_dict.keys())[0]], dtype='bool')
    nan_mask = ak.fill_none(nan_mask, False)
    for l in temp_dict:
        nan_mask = nan_mask & ~np.isnan(temp_dict[l])
        nan_mask = ak.fill_none(nan_mask, False)
    nanless = {}
    for l in array.fields:
        if array[l].fields == []:
            nanless[l] = temp_dict[l][nan_mask]
        else:
            for m in array[l].fields:
                nanless[m] = temp_dict[m][nan_mask]
    sizes = []
    for l in nanless:
        sizes.append(len(nanless[l]))
    losses = [(x - min(sizes)) for x in sizes]
    for l in nanless:
        nanless[l] = nanless[l][:min(sizes)]
    temp = pd.DataFrame.from_dict(nanless)
    array = ak.from_arrow(pa.Table.from_pandas(temp)) 
    return array