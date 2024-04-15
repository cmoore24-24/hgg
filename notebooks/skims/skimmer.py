import json
import dask
import dask_awkward as dak
import awkward as ak
import matplotlib.pyplot
from coffea import dataset_tools
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from ndcctools.taskvine import DaskVine
import time
import os
import warnings

full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    warnings.filterwarnings("ignore", module="coffea.*")
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
        fatjetSelect = (
            (events.FatJet.pt > 170)
            # & (abs(events.FatJet.eta) < 2.5)
            # & (events.FatJet.mass > 50)
            # & (events.FatJet.mass < 200)
        )
        events = events[
            ak.any(fatjetSelect, axis=1)
        ]
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
            f"/project01/ndcms/cmoore24/skims/stitch_test_skims/{dataset}",
            compute=False,
        )
        return skim_task

    tasks = dataset_tools.apply_to_fileset(
        analysis,
        #dataset_tools.slice_files(samples_ready, slice(None, 5)),
        samples_ready,
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
