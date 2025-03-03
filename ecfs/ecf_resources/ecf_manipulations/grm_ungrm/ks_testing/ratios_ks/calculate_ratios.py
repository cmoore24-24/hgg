import numpy as np
import awkward as ak
import warnings
import matplotlib.pyplot as plt
import hist
import math
import os
import json
import gc
import dask_awkward as dak
import argparse

warnings.filterwarnings('ignore', 'invalid value')
warnings.filterwarnings('ignore', 'divide by')
warnings.filterwarnings('ignore', 'overflow encountered')
warnings.filterwarnings('ignore', 'Conversion of an array')

parser = argparse.ArgumentParser(description="Script to process ECF ratios")
parser.add_argument('--proc_id', type=int, required=True, help="Process ID")
parser.add_argument('--end', type=int, required=True, help="End index for processing")
args = parser.parse_args()

proc_id = args.proc_id
end = args.end

path = '/scratch365/cmoore24/training/hgg/batch2024/ml_results_checking'
with open('./event_totals.json', 'r') as f:
    totals = json.load(f)

with open(f'{path}/my_xsecs.json', 'r') as f:
    xsecs = json.load(f)

xsecs2 = {}
for i in xsecs:
    if type(xsecs[i]) == dict:
        for j in xsecs[i]:
            xsecs2[j] = xsecs[i][j]
    else:
        xsecs2[i] = xsecs[i]
xsecs = xsecs2

totals2 = {}
for i in totals:
    if type(totals[i]) == dict:
        for j in totals[i]:
            totals2[f'{i}_{j}'] = totals[i][j]
    else:
        totals2[i] = totals[i]
totals = totals2

def read_files(input_path):
    if ('.parquet' in os.listdir(input_path)[0]):
        output = ak.from_parquet(f'{input_path}/new*', columns=columns)
    else:
        output = {}
        for i in os.listdir(input_path):
            if ('flat400' not in i):
                output[i] = ak.from_parquet(f'{input_path}/{i}/new*', columns=columns)
            else:
                continue
    return output

totals['ww'] = totals['diboson_ww']
totals['wz'] = totals['diboson_wz']
totals['zz'] = totals['diboson_zz']

region = 'nolepton'
path = '/project01/ndcms/cmoore24/skims/full_skims'

def add_ratio(ratio, dataframe):
    dash = ratio.find('/')
    asterisk = ratio.find('*')
    numerator = ratio[:dash]
    denominator = ratio[dash+1:asterisk]
    exponent = float(ratio[asterisk+2:])
    num_ecf = dataframe[numerator]
    den_ecf = dataframe[denominator]
    ecf_ratio = (num_ecf / (den_ecf**exponent))   
    return ecf_ratio

def get_num_den(ratio):
    dash = ratio.find('/')
    asterisk = ratio.find('*')
    numerator = ratio[:dash]
    denominator = ratio[dash+1:asterisk]  
    return (numerator, denominator)

def firsts(mc):
    for i in mc:
        if type(mc[i]) == dict:
            for j in mc[i]:
                for k in mc[i][j].fields:
                    if 'event' in k:
                        continue
                    else:
                        try:
                            mc[i][j][k] = ak.firsts(mc[i][j][k])
                        except:
                            continue
        else:
            for j in mc[i].fields:
                if 'event' in j:
                    continue
                else:
                    try:
                        mc[i][j] = ak.firsts(mc[i][j])
                    except:
                        continue
    return mc

ecf_list = dak.from_parquet('/project01/ndcms/cmoore24/skims/full_skims/nolepton/mc/hgg/new*').ungroomed_ecfs.fields

ratios = []
for i in range(len(ecf_list)):
    if ecf_list[i][2] == 'e':
        n1 = int(ecf_list[i][3])
        a = int(ecf_list[i][:2])
    else:
        n1 = int(ecf_list[i][2])
        a = int(ecf_list[i][0])
    for j in range(len(ecf_list)):
        if ecf_list[i] == ecf_list[j]:
            continue
        if ecf_list[j][2] == 'e':
            n2 = int(ecf_list[j][3])
            b = int(ecf_list[j][:2])
        else:
            n2 = int(ecf_list[j][2])
            b = int(ecf_list[j][0])
        if n1 < n2:
            continue
        else:
            beta1 = float(ecf_list[i][-2:])
            beta2 = float(ecf_list[j][-2:])
            exponent = (a * beta1) / (b * beta2)
            ratios.append(f'{ecf_list[i]}/{ecf_list[j]}**{exponent}')

upper = 1200
lower = 500
IL = 44.99

def ecf_hist(dataset, ecf_min, ecf_max):
    make_hist = hist.Hist.new.Reg(40, ecf_min, ecf_max, name='ECF', label='MC ECF').Weight()
    make_hist.fill(ECF=dataset)
    return make_hist

ecfs_ks = {}
with open(f'./outputs/ungroomed_ratio_{proc_id}.json','w') as f:
    json.dump(ecfs_ks, f)

if end >= len(ratios):
    end = len(ratios)
else:
    continue
    
for k in range(end-500, end):

    numerator, denominator = get_num_den(ratios[k])
    
    columns=['goodjets.msoftdrop', 'goodjets.pt', ("ungroomed_ecfs", f'{numerator}'), ("ungroomed_ecfs", f'{denominator}')]
    mc = read_files(f'{path}/{region}/mc')
    mc['ww'] = mc['diboson_ww']
    mc['wz'] = mc['diboson_wz']
    mc['zz'] = mc['diboson_zz']
    del(mc['diboson_ww'])
    del(mc['diboson_wz'])
    del(mc['diboson_zz'])
    mc = firsts(mc)
    data = read_files(f'{path}/{region}/data')
    data = firsts(data)

    for i in xsecs:
        if type(mc[i]) == dict:
            for j in mc[i]:
                mask = ((mc[i][j].goodjets.pt >= lower) & (mc[i][j].goodjets.pt <= upper))
                mc[i][j] = mc[i][j][mask]
        else:
            mask = ((mc[i].goodjets.pt >= lower) & (mc[i].goodjets.pt <= upper))
            mc[i] = mc[i][mask]

    for i in data:
        if type(data[i]) == dict:
            for j in data[i]:
                mask = ((data[i][j].goodjets.pt >= lower) & (data[i][j].goodjets.pt <= upper))
                data[i][j] = data[i][j][mask]
        else:
            mask = ((data[i].goodjets.pt >= lower) & (data[i].goodjets.pt <= upper))
            data[i] = data[i][mask]

    data_s = {}
    for i in data:
        if "Jet" in i:
            data_s[i] = data[i]  
    data_arr = ak.concatenate([data[i] for i in data_s])

    data_ratio = add_ratio(ratios[k], data_arr.ungroomed_ecfs)

    ratio_max = ak.max(data_ratio)
    ratio_min = ak.min(data_ratio)

    data_hist = hist.Hist.new.Reg(40, ratio_min, ratio_max, name='Ratio', label='Data Ratio').Weight()
    data_hist.fill(Ratio=data_ratio)

    mc2 = {}
    for i in xsecs:
        if type(mc[i]) == dict:
            for j in mc[i]:
                mc2[j] = mc[i][j]
        else:
            mc2[i] = mc[i]
    mc = mc2    

    mc_ratios = {}
    for i in mc:
        if type(mc[i]) == dict:
            for j in mc[i]:
                mc_ratios[j] = add_ratio(ratios[k], mc[i][j].ungroomed_ecfs)
        else:
            mc_ratios[i] = add_ratio(ratios[k], mc[i].ungroomed_ecfs)

    hists = {}
    for i in mc:
        if type(mc[i]) == dict:
            hists[i] = {}
            for j in mc[i]:
                hists[i][j] = ecf_hist(mc_ratios[i][j], ratio_min, ratio_max)
        else:
            hists[i] = ecf_hist(mc_ratios[i], ratio_min, ratio_max)

    scaleHgg = ((IL*(xsecs['hgg']*1000)*0.0817)/(totals['hgg']))
    hists['hgg'].view(flow=True)[:] *= scaleHgg
    
    scaleHbb = ((IL*(xsecs['hbb']*1000)*0.581)/(totals['hbb']))
    hists['hbb'].view(flow=True)[:] *= scaleHbb

    for i in mc:
        if (i == 'hgg') or (i == 'hbb'):
            continue
        else:
            scale = ((IL*(xsecs[i]*1000))/(totals[i]))
            hists[i].view(flow=True)[:] *= scale

    mc_hist = sum(hists[i] for i in hists)

    mc_values, mc_bins = mc_hist.to_numpy()
    data_values, data_bins = data_hist.to_numpy()
    mc_density = mc_values / mc_values.sum()
    data_density = data_values / data_values.sum()
    mc_cdf = np.cumsum(mc_density)
    data_cdf = np.cumsum(data_density)
    ks_statistic = np.max(np.abs(mc_cdf - data_cdf))
    try:
        adjusted = -math.log10(ks_statistic)

        if adjusted >= 3:
            ratio_max = np.nanpercentile(np.array(data_ratio), 95)
            ratio_min = ak.min(data_ratio)
        
            data_hist = hist.Hist.new.Reg(40, ratio_min, ratio_max, name='Ratio', label='Data Ratio').Weight()
            data_hist.fill(Ratio=data_ratio)

            hists = {}
            for i in mc:
                if type(mc[i]) == dict:
                    hists[i] = {}
                    for j in mc[i]:
                        hists[i][j] = ecf_hist(mc_ratios[i][j], ratio_min, ratio_max, 40)
                else:
                    hists[i] = ecf_hist(mc_ratios[i], ratio_min, ratio_max, 40)

            scaleHgg = ((IL*(xsecs['hgg']*1000)*0.0817)/(totals['hgg']))
            hists['hgg'].view(flow=True)[:] *= scaleHgg
            
            scaleHbb = ((IL*(xsecs['hbb']*1000)*0.581)/(totals['hbb']))
            hists['hbb'].view(flow=True)[:] *= scaleHbb
        
            for i in mc:
                if (i == 'hgg') or (i == 'hbb'):
                    continue
                else:
                    scale = ((IL*(xsecs[i]*1000))/(totals[i]))
                    hists[i].view(flow=True)[:] *= scale
        
            mc_hist = sum(hists[i] for i in hists)
        
            mc_values, mc_bins = mc_hist.to_numpy()
            data_values, data_bins = data_hist.to_numpy()
            mc_density = mc_values / mc_values.sum()
            data_density = data_values / data_values.sum()
            mc_cdf = np.cumsum(mc_density)
            data_cdf = np.cumsum(data_density)
            ks_statistic = np.max(np.abs(mc_cdf - data_cdf))
            adjusted = -math.log10(ks_statistic)

            with open(f'./outputs/ungroomed_ratio_{proc_id}.json','r') as f:
                ecf_ks = json.load(f)
        
            ecf_ks[ratios[k]] = adjusted
        
            with open(f'./outputs/ungroomed_ratio_{proc_id}.json','w') as f:
                json.dump(ecf_ks, f)

        else:
            with open(f'./outputs/ungroomed_ratio_{proc_id}.json','r') as f:
                ecf_ks = json.load(f)
        
            ecf_ks[ratios[k]] = adjusted
        
            with open(f'./outputs/ungroomed_ratio_{proc_id}.json','w') as f:
                json.dump(ecf_ks, f)
    except:
        with open(f'./outputs/ungroomed_ratio_{proc_id}.json','r') as f:
            ecf_ks = json.load(f)
        
        ecf_ks[ratios[k]] = None
        
        with open(f'./outputs/ungroomed_ratio_{proc_id}.json','w') as f:
            json.dump(ecf_ks, f)

    #print(f'{var} is done')
    gc.collect()
    print(k)