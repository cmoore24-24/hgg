import pandas as pd
import os
import time
import dask
import dask.dataframe as ddf
from ndcctools.taskvine import DaskVine
import sys
import warnings
import pickle


vars = [
    'area',
    'eta',
    'mass',
    'msoftdrop',
    'n2b1',
    'n3b1',
    'phi',
    'pt',
    'nConstituents',
    'color_ring',
    # 'd2b1',
    # 'u1',
    # 'u2',
    # 'u3',
    # 'd3',
    # 'm2',
    # 'm3',
    # 'n4'
    
]

desired_vars = []
for i in vars:
    fastparquet_comply = 'FatJets.' + i
    desired_vars.append(fastparquet_comply)
print(f'Variable output will be {desired_vars}')


warnings.filterwarnings('ignore', 'The fastparquet')

if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    qcd_paths = {
        '300to470':"/project01/ndcms/cmoore24/skims/fatjet_skims/300to470/",
        '470to600':"/project01/ndcms/cmoore24/skims/fatjet_skims/470to600/",
        '600to800':"/project01/ndcms/cmoore24/skims/fatjet_skims/600to800/",
        '800to1000':"/project01/ndcms/cmoore24/skims/fatjet_skims/800to1000/",
        '1000to1400':"/project01/ndcms/cmoore24/skims/fatjet_skims/1000to1400/",
        '1400to1800':"/project01/ndcms/cmoore24/skims/fatjet_skims/1400to1800/",
        '1800to2400':"/project01/ndcms/cmoore24/skims/fatjet_skims/1800to2400/",
        '2400to3200':"/project01/ndcms/cmoore24/skims/fatjet_skims/2400to3200/",
        '3200toInf':"/project01/ndcms/cmoore24/skims/fatjet_skims/3200toInf/"
    }

    signal_paths = {
        'hgg':"/project01/ndcms/cmoore24/skims/fatjet_skims/hgg/",
        'hbb':"/project01/ndcms/cmoore24/skims/fatjet_skims/hbb/"
    }


    super_start = time.time()
    qcd_dask = {}
    for i in list(qcd_paths.keys()):
        start = time.time()
        path = qcd_paths[i]
        filelist = os.listdir(path)
        qcd_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    qcd_stop = time.time()
    print(f"QCD compilation complete: {(qcd_stop-super_start)/60:.2f} minutes")

    signal_start = time.time()
    signal_dask = {}
    for i in list(signal_paths.keys()):
        start = time.time()
        path = signal_paths[i]
        filelist = os.listdir(path)
        signal_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    super_stop = time.time()
    print(f"Signal compilation complete: {(super_stop-signal_start)/60:.2f} minutes")

    print(f"Full compilation complete: {(super_stop-super_start)/60:.2f} minutes")
    
    qcd_compute_start = time.time()
    print('QCD Compute Start')
    output = dask.compute(
        qcd_dask,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]
    qcd_compute_stop = time.time()
    for i in output:
        output[i].to_parquet(f'parquet/{i}.parquet', engine='fastparquet')
    del(output)
    print(f"QCD Compute Complete: {(qcd_compute_stop-qcd_compute_start)/60:.2f} minutes")

    signal_compute_start = time.time()
    print('Signal Compute Start')
    output = dask.compute(
        signal_dask,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]
    for i in output:
        output[i].to_parquet(f'parquet/{i}.parquet', engine='fastparquet')
    signal_compute_stop = time.time()
    print(f"Signal Compute Complete: {(signal_compute_stop-signal_compute_start)/60:.2f} minutes")
    
    print(f'All done! {(signal_compute_stop-super_start)/60:.2f} minutes')