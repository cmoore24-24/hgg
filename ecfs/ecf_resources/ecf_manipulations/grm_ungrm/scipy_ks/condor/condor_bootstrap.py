import numpy as np
import awkward as ak
import warnings
import hist
import math
import os
import json
import gc
import dask_awkward as dak
from scipy.stats import kstest
import argparse
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, required=True, help="Index of the ratios list to use")
args = parser.parse_args()

index = args.index + 16000
print(f'The index is {index}')

warnings.filterwarnings('ignore', 'invalid value')
warnings.filterwarnings('ignore', 'divide by')
warnings.filterwarnings('ignore', 'overflow encountered')
warnings.filterwarnings('ignore', 'Conversion of an array')

path = '/scratch365/cmoore24/training/hgg/batch2024/ml_results_checking'
with open(f'{path}/subregion_event_totals.json', 'r') as f:
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
        output = ak.from_parquet(f'{input_path}/keep*', columns=columns)
    else:
        output = {}
        for i in os.listdir(input_path):
            if ('flat400' not in i):
                output[i] = ak.from_parquet(f'{input_path}/{i}/keep*', columns=columns)
            else:
                continue
    return output

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

ecf_list = dak.from_parquet('/project01/ndcms/cmoore24/skims/full_skims/nolepton/mc/2017/hgg/keep*').groomed_ecfs.fields

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

upper = 1000
lower = 475
IL = 44.99

def ecf_hist(dataset, ecf_min, ecf_max):
    make_hist = hist.Hist.new.Reg(40, ecf_min, ecf_max, name='ECF', label='MC ECF').Weight()
    make_hist.fill(ECF=dataset)
    return make_hist

ratio = ratios[index]

groom_choice = 'ungroomed'

numerator, denominator = get_num_den(ratio)

path = '/project01/ndcms/cmoore24/skims/full_skims/trijet/mc/2017/'
columns=['goodjets.msoftdrop', 'goodjets.pt', 'goodjets.trigger_mask', 
         (f"{groom_choice}_ecfs", f'{numerator}'), (f"{groom_choice}_ecfs", f'{denominator}')]
mc = read_files(f'{path}')

mc['ww'] = mc['diboson_ww']
mc['wz'] = mc['diboson_wz']
mc['zz'] = mc['diboson_zz']

del(mc['diboson_ww'])
del(mc['diboson_wz'])
del(mc['diboson_zz'])

mc = firsts(mc)

path = '/project01/ndcms/cmoore24/skims/full_skims/trijet/data/2017/'
data = read_files(f'{path}')
data = firsts(data)

for i in xsecs:
    try:
        if type(mc[i]) == dict:
            for j in mc[i]:
                mask = ((mc[i][j].goodjets.pt >= lower) & (mc[i][j].goodjets.pt <= upper) &
                        (mc[i][j].goodjets.msoftdrop >= 80) & (mc[i][j].goodjets.msoftdrop <= 170)
                       & (mc[i][j].goodjets.trigger_mask) )
                mc[i][j] = mc[i][j][mask]
        else:
            mask = ((mc[i].goodjets.pt >= lower) & (mc[i].goodjets.pt <= upper) &
                        (mc[i].goodjets.msoftdrop >= 80) & (mc[i].goodjets.msoftdrop <= 170)
                    & (mc[i].goodjets.trigger_mask))
            mc[i] = mc[i][mask]
    except:
        continue

for i in data:
    if type(data[i]) == dict:
        for j in data[i]:
            mask = ((data[i][j].goodjets.pt >= lower) & (data[i][j].goodjets.pt <= upper) &
                        (data[i][j].goodjets.msoftdrop >= 80) & (data[i][j].goodjets.msoftdrop <= 170)
                    & (data[i][j].goodjets.trigger_mask))
            data[i][j] = data[i][j][mask]
    else:
        mask = ((data[i].goodjets.pt >= lower) & (data[i].goodjets.pt <= upper) &
                        (data[i].goodjets.msoftdrop >= 80) & (data[i].goodjets.msoftdrop <= 170)
                 & (data[i].goodjets.trigger_mask))
        data[i] = data[i][mask]

data_s = {}
for i in data:
    if "Jet" in i:
        data_s[i] = data[i]  
data_arr = ak.concatenate([data[i] for i in data_s])

data_ratio = add_ratio(ratio, data_arr[f'{groom_choice}_ecfs'])

del(data_arr)
del(data)

mc_ratios = {}
for i in mc:
    if type(mc[i]) == dict:
        for j in mc[i]:
            mc_ratios[j] = add_ratio(ratio, mc[i][j][f'{groom_choice}_ecfs'])
    else:
        mc_ratios[i] = add_ratio(ratio, mc[i][f'{groom_choice}_ecfs'])

del(mc)

data_ratio = data_ratio[~ak.is_none(ak.nan_to_none(data_ratio))]

for i in mc_ratios:
    mc_ratios[i] = mc_ratios[i][~ak.is_none(ak.nan_to_none(mc_ratios[i]))]

def build_weighted_ks_input(mc, data, xsecs, totals, lumi):
    sim_values = []
    sim_weights = []
    
    for sample, arr in mc.items():
        if len(arr) == 0 or sample not in xsecs or sample not in totals:
            continue
        
        xsec = xsecs[sample]
        nevt = totals[sample]
        weight = (xsec * lumi) / nevt

        score_vals = ak.to_numpy(arr)
        sim_values.append(score_vals)
        sim_weights.append(np.full(len(score_vals), weight))


    sim_values = np.concatenate(sim_values)
    sim_weights = np.concatenate(sim_weights)

    data_values = []

    vals = ak.to_numpy(data)
    data_values.append(vals)
    data_values = np.concatenate(data_values)

    return data_values, sim_values, sim_weights

def make_weighted_ecdf(values, weights):
    sorter = np.argsort(values)
    sorted_vals = values[sorter]
    sorted_weights = weights[sorter]
    cumsum = np.cumsum(sorted_weights)
    cdf_vals = cumsum / cumsum[-1]

    def cdf_func(x):
        return np.interp(x, sorted_vals, cdf_vals, left=0.0, right=1.0)

    return cdf_func

def sample_from_weighted_cdf(values, weights, size):
    weights = weights / np.sum(weights)
    return np.random.choice(values, size=size, replace=True, p=weights)

data_vals, sim_vals, sim_wts = build_weighted_ks_input(mc_ratios, data_ratio, xsecs, totals, IL)
cdf_sim = make_weighted_ecdf(sim_vals, sim_wts)
observed_stat = kstest(data_vals, cdf=cdf_sim).statistic

# Bootstrap
n_bootstrap = 1000
boot_stats = []
for _ in trange(n_bootstrap, desc="Bootstrapping KS statistics"):
    sample = sample_from_weighted_cdf(sim_vals, sim_wts, size=len(data_vals))
    cdf_sample = make_weighted_ecdf(sample, np.ones_like(sample))
    stat = kstest(data_vals, cdf=cdf_sample).statistic
    boot_stats.append(stat)

boot_stats = np.array(boot_stats)
p_value = np.sum(boot_stats >= observed_stat) / len(boot_stats)

results_dict = {f'{ratio}':{'ks_statistic':observed_stat,
                'bootstrap_p': p_value,
                'adjusted_ks': -math.log10(observed_stat)}                   
                }

with open(f'/users/cmoore24/Public/hgg/ecfs/ecf_resources/ecf_manipulations/grm_ungrm/scipy_ks/results/{groom_choice}_{index}.json','w') as f:
    json.dump(results_dict, f)