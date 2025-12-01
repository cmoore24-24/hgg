import os
import awkward as ak
import math
from coffea.nanoevents.methods.nanoaod import behavior
import argparse
import numpy as np
import subprocess
import gc

parser = argparse.ArgumentParser(description="BDT Inference Script")
parser.add_argument('--path', type=str, required=True, help='File path')
parser.add_argument('--section', type=int, required=True, help='Which portion of the filelist to read')
args = parser.parse_args()
dir_name = args.path
section = args.section

field_list = ['goodjets',
            'ungroomed_ecfs',
            'groomed_ecfs',
            'event',
            # 'pnet_vals',
            'GenJetAK8',
            'GenPart',
            'SubJet',
            'Jet',
            'fixedGridRhoFastjetAll',
            'GenJet',
            'MET',
            'genWeight',
            'PSWeight',
            'Pileup',
            'matchedBoson',
            'L1PreFiringWeight',
            'run',
            'luminosityBlock',
            'LHEScaleWeight',
            'LHEPdfWeight']


for j in os.listdir(dir_name):
    print(f'Doing {j}')
    files = os.listdir(f'{dir_name}/{j}')
    files = [i for i in files if 'part' in i]
    if files == []:
        continue
    to_read = np.array_split(files, 24)[section].tolist()
    paths = [(dir_name + f'/{j}/{i}') for i in  to_read]
    events = ak.from_parquet(paths, behavior=behavior, columns=field_list)
    # events = ak.from_parquet(paths)
    params = ak.parameters(events) or {}
    record = params.get("__record__", "Events")
    params = {k: v for k, v in params.items() if k != "__record__"}

    print('Here1')
    
    
    field_dict = {}
    for i in field_list:
        if i in events.fields:
            field_dict = field_dict | {i:events[i]}

    print('Here2')

    skim = ak.zip(
            field_dict,
            with_name=record,
            parameters=params,
            depth_limit=1,
        )

    print('Here3')
    
    ak.to_parquet(skim, f'{dir_name}/{j}/reduced{section}.parquet', extensionarray=True)
    # del(skim)
    # del(events)
    gc.collect()
