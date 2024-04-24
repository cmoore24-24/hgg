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
    'pt',
    'color_ring',
    # 'eta',
    # 'phi',
    # 'n2b1',
    # 'n3b1',
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

    
    #Reminder: When Hgg and Hbb samples are implemented, turn this into a for loop
    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/300to470/"
    super_start = time.time()
    filelist = os.listdir(path)
    dask_q347 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q347 compiled, {len(filelist)} files: {(stop-super_start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/470to600/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q476 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q476 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/600to800/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q68 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q68 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/800to1000/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q810 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q810 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/1000to1400/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q1014 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q1014 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/1400to1800/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q1418 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q1418 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/1800to2400/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q1824 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q1824 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/2400to3200/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q2432 = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    stop = time.time()
    print(f"dask_q2432 compiled, {len(filelist)} files: {(stop-start)/60:.2f} minutes")

    path = "/project01/ndcms/cmoore24/skims/stitch_test_skims/3200toInf/"
    start = time.time()
    filelist = os.listdir(path)
    dask_q32inf = ddf.concat([ddf.read_parquet(f"{path}{f}", engine='fastparquet') for f in filelist])
    super_stop = time.time()
    print(f"dask_q32inf compiled, {len(filelist)} files: {(super_stop-start)/60:.2f} minutes")

    print(f"Full compilation complete: {(super_stop-super_start)/60:.2f} minutes")

    dask_dict = {}
    dask_dict['q347'] = dask_q347[desired_vars]
    dask_dict['q476'] = dask_q476[desired_vars]
    dask_dict['q68'] = dask_q68[desired_vars]
    dask_dict['q810'] = dask_q810[desired_vars]
    dask_dict['q1014'] = dask_q1014[desired_vars]
    dask_dict['q1418'] = dask_q1418[desired_vars]
    dask_dict['q1824'] = dask_q1824[desired_vars]
    dask_dict['q2432'] = dask_q2432[desired_vars]
    dask_dict['q32inf'] = dask_q32inf[desired_vars]
    
    compute_start = time.time()
    print('Compute Start')
    output = dask.compute(
        dask_dict,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )[0]
    compute_stop = time.time()
    print(f"{sys.getsizeof(output)} MB")
    print(f"Compute Complete: {(compute_stop-compute_start)/60:.2f} minutes")
    
    print(f'All done! {(compute_stop-super_start)/60:.2f} minutes')
    with open('./qcd_pt_cr.pkl', 'wb') as f:
        pickle.dump(output, f)