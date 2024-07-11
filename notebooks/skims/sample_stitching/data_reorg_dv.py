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
    'd2b1',
    'u1',
    'u2',
    'u3',
    'd3',
    'm2',
    'm3',
    'n4',
    'particleNet_HbbvsQCD',
    'particleNetMD_QCD'
    
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

    # qcd_paths = {
    #     'qcd_300to470':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_300to470/",
    #     'qcd_470to600':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_470to600/",
    #     'qcd_600to800':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_600to800/",
    #     'qcd_800to1000':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_800to1000/",
    #     'qcd_1000to1400':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_1000to1400/",
    #     'qcd_1400to1800':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_1400to1800/",
    #     'qcd_1800to2400':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_1800to2400/",
    #     'qcd_2400to3200':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_2400to3200/",
    #     'qcd_3200toInf':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/qcd/qcd_3200toInf/"
    # }

    diboson_paths = {
        'ww':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/diboson/diboson_ww/",
        'zz':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/diboson/diboson_zz/",
        'wz':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/diboson/diboson_wz/",
    }

    # singletop_path = {
    #     'singletop':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/singletop/",
    # }

    # ttboosted_paths = {
    #     'ttboosted_700to1000':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/ttboosted/ttboosted_700to1000/",
    #     'ttboosted_1000toInf':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/ttboosted/ttboosted_1000toInf/",
    # }

    # wqq_paths = {
    #     'wqq_200to400':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/wqq/wqq_200to400/",
    #     'wqq_400to600':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/wqq/wqq_400to600/",
    #     'wqq_600to800':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/wqq/wqq_600to800/",
    #     'wqq_800toInf':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/wqq/wqq_800toInf/",
    # }

    # zqq_paths = {
    #     'zqq_200to400':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/zqq/zqq_200to400/",
    #     'zqq_400to600':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/zqq/zqq_400to600/",
    #     'zqq_600to800':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/zqq/zqq_600to800/",
    #     'zqq_800toInf':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/zqq/zqq_800toInf/",
    # }

    # signal_paths = {
    #     'hgg':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/hgg/",
    #     'hbb':"/project01/ndcms/cmoore24/skims/fatjet_skims/no_nsub_cut/hbb/"
    # }


    super_start = time.time()
    # qcd_dask = {}
    # for i in list(qcd_paths.keys()):
    #     start = time.time()
    #     path = qcd_paths[i]
    #     filelist = os.listdir(path)
    #     qcd_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
    #     stop = time.time()
    #     print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    # qcd_stop = time.time()
    # print(f"QCD compilation complete: {(qcd_stop-super_start)/60:.2f} minutes")
    # print('\n')

    # signal_start = time.time()
    # signal_dask = {}
    # for i in list(signal_paths.keys()):
    #     start = time.time()
    #     path = signal_paths[i]
    #     filelist = os.listdir(path)
    #     signal_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
    #     stop = time.time()
    #     print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    # signal_stop = time.time()
    # print(f"Signal compilation complete: {(signal_stop-signal_start)/60:.2f} minutes")
    # print('\n')

    diboson_start = time.time()
    diboson_dask = {}
    for i in list(diboson_paths.keys()):
        start = time.time()
        path = diboson_paths[i]
        filelist = os.listdir(path)
        diboson_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    diboson_stop = time.time()
    super_stop = time.time()
    print(f"Diboson compilation complete: {(diboson_stop-diboson_start)/60:.2f} minutes")
    print('\n')

    # singletop_start = time.time()
    # singletop_dask = {}
    # for i in list(singletop_path.keys()):
    #     start = time.time()
    #     path = singletop_path[i]
    #     filelist = os.listdir(path)
    #     singletop_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
    #     stop = time.time()
    #     print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    # singletop_stop = time.time()
    # print(f"Singletop compilation complete: {(singletop_stop-singletop_start)/60:.2f} minutes")
    # print('\n')

    # ttboosted_start = time.time()
    # ttboosted_dask = {}
    # for i in list(ttboosted_paths.keys()):
    #     start = time.time()
    #     path = ttboosted_paths[i]
    #     filelist = os.listdir(path)
    #     ttboosted_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
    #     stop = time.time()
    #     print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    # ttboosted_stop = time.time()
    # print(f"TTBoosted compilation complete: {(ttboosted_stop-ttboosted_start)/60:.2f} minutes")
    # print('\n')

    # wqq_start = time.time()
    # wqq_dask = {}
    # for i in list(wqq_paths.keys()):
    #     start = time.time()
    #     path = wqq_paths[i]
    #     filelist = os.listdir(path)
    #     wqq_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
    #     stop = time.time()
    #     print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    # wqq_stop = time.time()
    # print(f"Wqq compilation complete: {(wqq_stop-wqq_start)/60:.2f} minutes")
    # print('\n')

    # zqq_start = time.time()
    # zqq_dask = {}
    # for i in list(zqq_paths.keys()):
    #     start = time.time()
    #     path = zqq_paths[i]
    #     filelist = os.listdir(path)
    #     zqq_dask[i] = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])[desired_vars]
    #     stop = time.time()
    #     print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    # super_stop = time.time()
    # print(f"Zqq compilation complete: {(super_stop-zqq_start)/60:.2f} minutes")
    # print('\n')

    print(f"Full compilation complete: {(super_stop-super_start)/60:.2f} minutes")
    print('\n')
    
    # qcd_compute_start = time.time()
    # print('QCD Compute Start')
    # print('\n')
    # output = dask.compute(
    #     qcd_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # qcd_compute_stop = time.time()
    # for i in output:
    #     output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    # del(output)
    # print(f"QCD Compute Complete: {(qcd_compute_stop-qcd_compute_start)/60:.2f} minutes")
    # print('\n')

    diboson_compute_start = time.time()
    print('Diboson Compute Start')
    output = dask.compute(
        diboson_dask,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]
    diboson_compute_stop = time.time()
    for i in output:
        output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    del(output)
    print(f"Diboson Compute Complete: {(diboson_compute_stop-diboson_compute_start)/60:.2f} minutes")
    print('\n')

    # singletop_compute_start = time.time()
    # print('Singletop Compute Start')
    # output = dask.compute(
    #     singletop_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # singletop_compute_stop = time.time()
    # for i in output:
    #     output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    # del(output)
    # print(f"Singletop Compute Complete: {(singletop_compute_stop-singletop_compute_start)/60:.2f} minutes")
    # print('\n')

    # ttboosted_compute_start = time.time()
    # print('ttboosted Compute Start')
    # output = dask.compute(
    #     ttboosted_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # ttboosted_compute_stop = time.time()
    # for i in output:
    #     output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    # del(output)
    # print(f"ttboosted Compute Complete: {(ttboosted_compute_stop-ttboosted_compute_start)/60:.2f} minutes")
    # print('\n')

    # wqq_compute_start = time.time()
    # print('Wqq Compute Start')
    # output = dask.compute(
    #     wqq_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # wqq_compute_stop = time.time()
    # for i in output:
    #     output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    # del(output)
    # print(f"Wqq Compute Complete: {(wqq_compute_stop-wqq_compute_start)/60:.2f} minutes")
    # print('\n')

    # zqq_compute_start = time.time()
    # print('Zqq Compute Start')
    # output = dask.compute(
    #     zqq_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # zqq_compute_stop = time.time()
    # for i in output:
    #     output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    # del(output)
    # print(f"Zqq Compute Complete: {(zqq_compute_stop-zqq_compute_start)/60:.2f} minutes")
    # print('\n')

    # signal_compute_start = time.time()
    # print('Signal Compute Start')
    # output = dask.compute(
    #     signal_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # for i in output:
    #     output[i].to_parquet(f'parquet/no_subcut/{i}.parquet', engine='fastparquet')
    # signal_compute_stop = time.time()
    # print(f"Signal Compute Complete: {(signal_compute_stop-signal_compute_start)/60:.2f} minutes")
    # print('\n')
    
    print(f'All done! {(signal_compute_stop-super_start)/60:.2f} minutes')