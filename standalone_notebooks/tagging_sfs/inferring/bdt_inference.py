import xgboost as xgb
import awkward as ak
import warnings
import pickle
import os
import json
import numpy as np
import argparse
from coffea.nanoevents.methods.nanoaod import behavior

parser = argparse.ArgumentParser(description="BDT Inference Script")
parser.add_argument('--path', type=str, required=True, help='File path')
parser.add_argument('--section', type=int, required=True, help='Which portion of the filelist to read')
args = parser.parse_args()
path = args.path
section = args.section

warnings.filterwarnings('ignore', 'invalid value')
warnings.filterwarnings('ignore', 'No format')
warnings.filterwarnings('ignore', 'overflow encountered in exp')

### hgg model machinery ###

def add_ratios(ratio, array):
    dash = ratio.find('/')
    asterisk = ratio.find('*')
    numerator = ratio[4:dash]
    denominator = ratio[dash+1:asterisk]
    exponent = float(ratio[asterisk+2:].replace('_','.'))
    if ratio[:3] == 'grm':
        num_ecf = array.groomed_ecfs[numerator]
        den_ecf = array.groomed_ecfs[denominator]
    elif ratio[:3] == 'ugm':
        num_ecf = array.ungroomed_ecfs[numerator]
        den_ecf = array.ungroomed_ecfs[denominator]
    ecf_ratio = (num_ecf / (den_ecf**exponent))   
    return ecf_ratio

ratios = 160

hgg_model_path = f'/scratch365/cmoore24/training/hgg/final_draft_dev/QCDgg/outputs/{ratios}'

hgg_bst = xgb.Booster()
hgg_bst.load_model(f'{hgg_model_path}/bdt_model.json')

scaler = f'{hgg_model_path}/scaler.pkl'
with open(scaler, 'rb') as f:
    scaler = pickle.load(f)

hgg_feature_names = hgg_bst.feature_names

### gluon model machinery ###

gluon_path = f'/scratch365/cmoore24/training/hgg/qg_final_draft/output/testing'
gluon_bst = xgb.Booster()
gluon_bst.load_model(f'{gluon_path}/bdt_model.json')

to_add = ['eta','lsf3']

def add_gluon_features(arr):
    training_dict = {}
    for i in to_add:
        training_dict[f'lead_{i}'] = arr.FatJet[i][:,0]
        training_dict[f'sub_{i}'] = arr.FatJet[i][:,1]
        training_dict[f'subsub_{i}'] = arr.FatJet[i][:,2]
    for i in arr.extra_vars.fields:
        if ak.any(np.isinf(arr.extra_vars[i])):
            training_dict[i] = ak.where(np.isinf(arr.extra_vars[i]), np.nan, arr.extra_vars[i])
        else:
            training_dict[i] = arr.extra_vars[i]
    arr['gluon_features'] = ak.zip(training_dict, depth_limit=1)
    return arr

gluon_features = gluon_bst.feature_names


### Do inferring ###

for j in os.listdir(path):
    print(f'Doing {j}')
    files = os.listdir(f'{path}/{j}')
    files = [i for i in files if 'reduced' in i]
    if files == []:
        continue
    to_read = np.array_split(files, 24)[section].tolist()
    paths = [(path + f'/{j}/{i}') for i in  to_read]
    file = ak.from_parquet(paths, behavior=behavior)

    ### hgg inference ###
    
    ecf_ratios = {}
    
    for ratio in hgg_feature_names:
        ecf_ratios[ratio] = add_ratios(ratio, file)
    
    ecf_ratios = ak.zip(ecf_ratios, depth_limit=1)
    
    file_np = np.column_stack([ak.to_numpy(ecf_ratios[feature]) for feature in hgg_feature_names])
    file_np = scaler.transform(file_np)
    file_xgb = xgb.DMatrix(file_np, feature_names=hgg_feature_names)
    file['goodjets', 'GSscore'] = hgg_bst.predict(file_xgb, 
                                                  # iteration_range=(0, hgg_bst.best_iteration + 1)
                                                 )
    # file['FatJet', 'GSscore'] = file['goodjets','GSscore']
    
    print('Inferred GSscore!')

    ### gluon inference ###

    file = add_gluon_features(file)
    gluon_np = np.column_stack([ak.to_numpy(file.gluon_features[feature]) for feature in gluon_features])
    gluon_xgb = xgb.DMatrix(gluon_np, feature_names=gluon_features)
    leading_pred = ak.unflatten(gluon_bst.predict(gluon_xgb, 
                                                  iteration_range=(0, gluon_bst.best_iteration + 1)
                                                 )
                                , counts=1)
    file['FatJet', 'GluonBDT'] = leading_pred
    file['goodjets', 'GluonBDT'] = leading_pred
    

    ak.to_parquet(file, f'{path}/{j}/keep{section}.parquet', extensionarray=True)
    