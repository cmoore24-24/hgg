import os
import time
import dask
import awkward as ak
import dask_awkward as dak
from ndcctools.taskvine import DaskVine


var = [
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
    'particleNetMD_QCD',
    'num_fatjets',
    
]



if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/'
    qcd_paths = {
        'qcd_300to470':f"{path}qcd_300to470/",
        'qcd_470to600':f"{path}qcd_470to600/",
        'qcd_600to800':f"{path}qcd_600to800/",
        'qcd_800to1000':f"{path}qcd_800to1000/",
        'qcd_1000to1400':f"{path}qcd_1000to1400/",
        'qcd_1400to1800':f"{path}qcd_1400to1800/",
        'qcd_1800to2400':f"{path}qcd_1800to2400/",
        'qcd_2400to3200':f"{path}qcd_2400to3200/",
        'qcd_3200toInf':f"{path}qcd_3200toInf/"
    }

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/'
    diboson_paths = {
        'ww':f"{path}diboson_ww/",
        'zz':f"{path}diboson_zz/",
        'wz':f"{path}diboson_wz/",
        'ww4q':f"{path}diboson_ww4q/",
    }

    singletop_path = {
        'singletop':"/project01/ndcms/cmoore24/skims/full_cut_skims/singletop",
    }

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/'
    ttboosted_paths = {
        'ttboosted_700to1000':f"{path}ttboosted_700to1000/",
        'ttboosted_1000toInf':f"{path}ttboosted_1000toInf/",
    }

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/'
    wqq_paths = {
        #'wqq_200to400':f"{path}wqq_200to400/",
        #'wqq_400to600':f"{path}wqq_400to600/",
        'wqq_600to800':f"{path}wqq_600to800/",
        'wqq_800toInf':f"{path}wqq_800toInf/",
    }

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/'
    zqq_paths = {
        #'zqq_200to400':f"{path}zqq_200to400/",
        #'zqq_400to600':f"{path}zqq_400to600/",
        #'zqq_600to800':f"{path}zqq_600to800/",
        'zqq_800toInf':f"{path}zqq_800toInf/",
    }

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/'
    signal_paths = {
        'hgg':f"{path}hgg/",
        'hbb':f"{path}hbb/"
    }


    super_start = time.time()
    qcd_dask = {}
    for i in list(qcd_paths.keys()):
        start = time.time()
        path = qcd_paths[i]
        filelist = os.listdir(path)
        qcd_dask[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    qcd_stop = time.time()
    print(f"QCD compilation complete: {(qcd_stop-super_start)/60:.2f} minutes")
    print('\n')

    signal_start = time.time()
    signal_dak = {}
    for i in list(signal_paths.keys()):
        start = time.time()
        path = signal_paths[i]
        filelist = os.listdir(path)
        signal_dak[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    signal_stop = time.time()
    print(f"Signal compilation complete: {(signal_stop-signal_start)/60:.2f} minutes")
    print('\n')

    diboson_start = time.time()
    diboson_dask = {}
    for i in list(diboson_paths.keys()):
        start = time.time()
        path = diboson_paths[i]
        filelist = os.listdir(path)
        diboson_dask[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    diboson_stop = time.time()
    print(f"Diboson compilation complete: {(diboson_stop-diboson_start)/60:.2f} minutes")
    print('\n')

    singletop_start = time.time()
    singletop_dask = {}
    for i in list(singletop_path.keys()):
        start = time.time()
        path = singletop_path[i]
        filelist = os.listdir(path)
        singletop_dask[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    singletop_stop = time.time()
    print(f"Singletop compilation complete: {(singletop_stop-singletop_start)/60:.2f} minutes")
    print('\n')

    ttboosted_start = time.time()
    ttboosted_dask = {}
    for i in list(ttboosted_paths.keys()):
        start = time.time()
        path = ttboosted_paths[i]
        filelist = os.listdir(path)
        ttboosted_dask[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    ttboosted_stop = time.time()
    print(f"TTBoosted compilation complete: {(ttboosted_stop-ttboosted_start)/60:.2f} minutes")
    print('\n')

    wqq_start = time.time()
    wqq_dask = {}
    for i in list(wqq_paths.keys()):
        start = time.time()
        path = wqq_paths[i]
        filelist = os.listdir(path)
        wqq_dask[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    wqq_stop = time.time()
    print(f"Wqq compilation complete: {(wqq_stop-wqq_start)/60:.2f} minutes")
    print('\n')

    zqq_start = time.time()
    zqq_dask = {}
    for i in list(zqq_paths.keys()):
        start = time.time()
        path = zqq_paths[i]
        filelist = os.listdir(path)
        zqq_dask[i] = dak.from_parquet(f"{path}")['goodjets'][var]
        stop = time.time()
        print(f"{i} compiled, {len(filelist)} files: {(stop - start)/60:.2f} minutes")
    zqq_stop = time.time()
    print(f"Zqq compilation complete: {(zqq_stop-zqq_start)/60:.2f} minutes")
    print('\n')

    super_stop = time.time()
    print(f"Full compilation complete: {(super_stop-super_start)/60:.2f} minutes")
    print('\n')

    path = '/project01/ndcms/cmoore24/skims/full_cut_skims/sole_vars'
    
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
    #     ak.to_parquet(output[i], f'{path}/qcd/{i}.parquet')
    # del(output)
    # print(f"QCD Compute Complete: {(qcd_compute_stop-qcd_compute_start)/60:.2f} minutes")
    # print('\n')

    # diboson_compute_start = time.time()
    # print('Diboson Compute Start')
    # output = dask.compute(
    #     diboson_dask,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    # diboson_compute_stop = time.time()
    # for i in output:
    #     ak.to_parquet(output[i], f'{path}/diboson/{i}.parquet')
    # del(output)
    # print(f"Diboson Compute Complete: {(diboson_compute_stop-diboson_compute_start)/60:.2f} minutes")
    # print('\n')

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
    #     ak.to_parquet(output[i], f'{path}/singletop/{i}.parquet')
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
    #     ak.to_parquet(output[i], f'{path}/ttboosted/{i}.parquet')
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
    #     ak.to_parquet(output[i], f'{path}/wqq/{i}.parquet')
    # del(output)
    # print(f"Wqq Compute Complete: {(wqq_compute_stop-wqq_compute_start)/60:.2f} minutes")
    # print('\n')

    zqq_compute_start = time.time()
    print('Zqq Compute Start')
    output = dask.compute(
        zqq_dask,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]
    zqq_compute_stop = time.time()
    for i in output:
        ak.to_parquet(output[i], f'{path}/zqq/{i}.parquet')
    del(output)
    print(f"Zqq Compute Complete: {(zqq_compute_stop-zqq_compute_start)/60:.2f} minutes")
    print('\n')

    signal_compute_start = time.time()
    print('Signal Compute Start')
    output = dask.compute(
        signal_dak,
        optimize_graph=False,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]
    for i in output:
        ak.to_parquet(output[i], f'{path}/signal/{i}.parquet')
    signal_compute_stop = time.time()
    print(f"Signal Compute Complete: {(signal_compute_stop-signal_compute_start)/60:.2f} minutes")
    print('\n')
    
    print(f'All done! {(signal_compute_stop-super_start)/60:.2f} minutes')