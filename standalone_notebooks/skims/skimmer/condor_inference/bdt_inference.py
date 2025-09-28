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

model_path = f'/scratch365/cmoore24/training/hgg/final_draft_dev/QCDgg/outputs/{ratios}'

bst = xgb.Booster()
bst.load_model(f'{model_path}/bdt_model.json')

scaler = f'{model_path}/scaler.pkl'
with open(scaler, 'rb') as f:
    scaler = pickle.load(f)

feature_names = bst.feature_names

for j in os.listdir(path):
    print(f'Doing {j}')
    files = os.listdir(f'{path}/{j}')
    files = [i for i in files if 'reduced' in i]
    if files == []:
        continue
    to_read = np.array_split(files, 24)[section].tolist()
    paths = [(path + f'/{j}/{i}') for i in  to_read]
    file = ak.from_parquet(paths, behavior=behavior)
    
    ecf_ratios = {}
    
    for i in feature_names:
        ecf_ratios[i] = add_ratios(i, file)
    
    ecf_ratios = ak.zip(ecf_ratios, depth_limit=1)
    
    file_np = np.column_stack([ak.to_numpy(ecf_ratios[feature]) for feature in feature_names])
    file_np = scaler.transform(file_np)
    file_xgb = xgb.DMatrix(file_np, feature_names=feature_names)
    file['goodjets', 'GSscore'] = bst.predict(file_xgb)
    
    print('Inferred!')

    ak.to_parquet(file, f'{path}/{j}/keep{section}.parquet', extensionarray=True)
    