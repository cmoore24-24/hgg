import json
import dask
import dask_awkward as dak
import awkward as ak
import numpy as np
from coffea import dataset_tools
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from ndcctools.taskvine import DaskVine
import fastjet
import time
import os
import warnings
from variable_functions import *

full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    m.tune("temp-replica-count", 3)
    
    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "invalid value")
    
    # with open('output_datasets.json', 'r') as f:
    #     samples = json.load(f)

    # print('doing samples')
    # sample_start = time.time()
    # samples_ready, samples = dataset_tools.preprocess(
    #     samples,
    #     step_size=50_000,
    #     skip_bad_files=True,
    #     recalculate_steps=True,
    #     save_form=False,
    # )
    # sample_stop = time.time()
    # print('samples done')
    # print('full sample time is ' + str((sample_stop - sample_start)/60))

    # with open("samples_ready.json", "w") as fout:
    #     json.dump(samples_ready, fout)

    with open("samples_ready.json", 'r') as fin:
        samples_ready = json.load(fin)

    def analysis(events):
        dataset = events.metadata["dataset"]


        num_sub = ak.unflatten(num_subjets(events.FatJet, cluster_val=0.4), counts=ak.num(events.FatJet))
        events['FatJet', 'num_subjets'] = num_sub
        
        fatjetSelect = (
            (events.FatJet.pt > 250)
            & (events.FatJet.num_subjets >= 3)
            & (abs(events.FatJet.eta) < 2.5)
            & (events.FatJet.mass > 50)
            & (events.FatJet.mass < 200)
        )
        events = events[
            ak.any(fatjetSelect, axis=1)
        ]

        cut_to_fix_softdrop = (ak.num(events.FatJet.constituents.pf, axis=2) > 0)
        events = events[ak.all(cut_to_fix_softdrop, axis=1)]

        events['PFCands', 'pt'] = (
                events.PFCands.pt
                * events.PFCands.puppiWeight
            )

        events['FatJet', 'color_ring'] = ak.unflatten(
             color_ring(events.FatJet, cluster_val=0.4), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'd2b1'] = ak.unflatten(
             d2_calc(events.FatJet), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'u1'] = ak.unflatten(
             u_calc(events.FatJet, 1), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'u2'] = ak.unflatten(
             u_calc(events.FatJet, 2), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'u3'] = ak.unflatten(
             u_calc(events.FatJet, 3), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'd3'] = ak.unflatten(
             d3_calc(events.FatJet), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'm2'] = ak.unflatten(
             m2_calc(events.FatJet), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'm3'] = ak.unflatten(
             m3_calc(events.FatJet), counts=ak.num(events.FatJet)
        )

        events['FatJet', 'n4'] = ak.unflatten(
             n4_calc(events.FatJet), counts=ak.num(events.FatJet)
        )
        
        skim = ak.zip(
            {
                "FatJets": ak.flatten(events.FatJet, axis=1),
                #  "MET": events.MET,
                #  "Photon": events.Photon,
                # "Subjets": events.SubJet,
                # "FJ_PFCands": events.FatJet.constituents.pf,
            },
            depth_limit=1,
        )
        
        skim_task = dak.to_parquet(
            skim,
            f"/project01/ndcms/cmoore24/skims/full_fatjet_skims/{dataset}",
            compute=False,
        )
        return skim_task

    subset = {}
    subset['3200toInf'] = samples_ready['3200toInf']
    tasks = dataset_tools.apply_to_fileset(
        analysis,
        #dataset_tools.slice_files(samples_ready, slice(None, 20)),
        #samples_ready,
        subset,
        uproot_options={"allow_read_errors_with_report": True},
        schemaclass = PFNanoAODSchema,
    )

    print('start compute')
    computed, report = dask.compute(
            *tasks,
            scheduler=m.get,
            resources={"cores": 1},
            resources_mode=None,
            lazy_transfers=True,
            #task_mode="function_calls",
            lib_resources={'cores': 12, 'slots': 12},
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))
