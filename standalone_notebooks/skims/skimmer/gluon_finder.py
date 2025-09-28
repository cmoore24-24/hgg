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
import scipy
import pickle
import subprocess
import argparse

# parser = argparse.ArgumentParser(description="Select dataset to process")
# parser.add_argument("--dataset", type=str, required=True, help="String to match in dataset names")
# args = parser.parse_args()

# skim_ds = args.dataset
num = '16'
index = f'qcd'
template_name = f'flat{num}'
samples_process = False
full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9101, 9200],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info/",
        run_info_template=f'{template_name}',
    )

    m.tune("temp-replica-count", 3)
    m.tune("worker-source-max-transfers", 100000)
    # m.tune("transfer-temps-recovery", 1)
    
    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "invalid value")

######## Uncomment the following section if you need to create an input_datasets.json for new datasets ########

    # samples_path = '/project01/ndcms/cmoore24/samples'  ## Change this path to where data is
    # filelist = {}
    # categories = os.listdir(samples_path)
    # print(categories)
    # for i in categories:
    #     if '.root' in os.listdir(f'{samples_path}/{i}')[0]:
    #         files = os.listdir(f'{samples_path}/{i}')
    #         filelist[i] = [f'{samples_path}/{i}/{file}' for file in files]
    #     else:
    #         sub_cats = os.listdir(f'{samples_path}/{i}')
    #         for j in sub_cats:
    #             if '.root' in os.listdir(f'{samples_path}/{i}/{j}')[0]:
    #                 files = os.listdir(f'{samples_path}/{i}/{j}')
    #                 filelist[f'{i}_{j}'] = [f'{samples_path}/{i}/{j}/{file}' for file in files]

    # input_dict = {}
    # for i in filelist:
    #     input_dict[i] = {}
    #     input_dict[i]['files'] = {}
    #     for j in filelist[i]:
    #         input_dict[i]['files'][j] = {'object_path': 'Events'}

    # with open('input_datasets.json', 'w') as fin:
    #     json.dump(dict, fin)


    ####### Uncomment the following section if you need to pre-process the datasets present in input_datasets.json ########
    if samples_process:
        with open('input_datasets.json', 'r') as f:
            samples = json.load(f)
    
        print('doing samples')
        sample_start = time.time()
    
        @dask.delayed
        def sampler(samples):
           samples_ready, samples = dataset_tools.preprocess(
               samples,
               step_size=50_000, ## Change this step size to adjust the size of chunks of events
               skip_bad_files=True,
               recalculate_steps=True,
               save_form=False,
           )
           return samples_ready
        
        sampler_dict = {}
        for i in samples:
           sampler_dict[i] = sampler(samples[i])
        
        print('Compute')
        samples_postprocess = dask.compute(
           sampler_dict,
           scheduler=m.get,
           resources={"cores": 1},
           # resources_mode=None,
           # prune_files=True,
           prune_depth=0,
           worker_transfers=True,
           task_mode="function-calls",
           lib_resources={'cores': 24, 'slots': 24},
        )[0]
        
        samples_ready = {}
        for i in samples_postprocess:
           samples_ready[i] = samples_postprocess[i]['files']
        
        sample_stop = time.time()
        print('samples done')
        print('full sample time is ' + str((sample_stop - sample_start)/60))
        
        with open("samples_ready.json", "w") as fout:
           json.dump(samples_ready, fout)

    ######## The analysis portion begins here ########
    
    with open("samples_ready.json", 'r') as f:
        samples_ready = json.load(f)

    with open('triggers.json', 'r') as f:
        triggers = json.load(f)

    def apply_selections(events, region, trigger, goodmuon, pdgid=None, is_wz=False):     
        fatjetSelect = (
            (events.FatJet.pt >= 475)
            & (events.FatJet.pt <= 1000)
            & (abs(events.FatJet.eta) <= 2.4)
            & (events.FatJet.msoftdrop >= 40)
            & (events.FatJet.msoftdrop <= 200)
            & (region)
            # & (ak.fill_none(events.FatJet.delta_r(events.FatJet.nearest(events.Muon[goodmuon], axis=1)) > 0.8, True))
            & (trigger)
            & (events.FatJet.btag_count == 0)
            & (ak.num(events.FatJet) >= 3)
        )
        
        if (pdgid != None) or (is_wz):
            if is_wz:
                genparts = events.GenPart[
                    ((abs(events.GenPart.pdgId) == 24)|(events.GenPart.pdgId == 23))
                    & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
                ]
            else:
                genparts = events.GenPart[
                    (abs(events.GenPart.pdgId) == pdgid)
                    & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
                ]
            parents = events.FatJet.nearest(genparts, threshold=0.2)
            matched_jets = ~ak.is_none(parents, axis=1)
            fatjetSelect = ((fatjetSelect) & (matched_jets))
        return fatjetSelect

    def analysis(events):
        dataset = events.metadata["dataset"]

        events['PFCands', 'pt'] = (
            events.PFCands.pt
            * events.PFCands.puppiWeight
        )
        
        cut_to_fix_softdrop = (ak.num(events.FatJet.constituents.pf, axis=2) > 0)
        events = events[ak.all(cut_to_fix_softdrop, axis=1)]

        trigger = ak.zeros_like(ak.firsts(events.FatJet.pt), dtype='bool')
        for t in triggers['2017']:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        trigger = ak.fill_none(trigger, False)

        # event.MET < 140 for regular boosted higgs, try > 50 to try focusing on W jets

        events['FatJet', 'num_fatjets'] = ak.num(events.FatJet)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25) # invert the isolation cut; > 0.25, check for QCD (maybe try > 1.0)
            & events.Muon.looseId
        )

        nmuons = ak.sum(goodmuon, axis=1)
        leadingmuon = ak.firsts(events.Muon[goodmuon])

        goodelectron = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= 2) #events.Electron.LOOSE
        )
        nelectrons = ak.sum(goodelectron, axis=1)

        ntaus = ak.sum(
            (
                (events.Tau.pt > 20)
                & (abs(events.Tau.eta) < 2.3)
                & (events.Tau.rawIso < 5)
                & (events.Tau.idDeepTau2017v2p1VSjet)
                & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
                & ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2)
            ),
            axis=1,
        )

        nolepton = ((nmuons == 0) & (nelectrons == 0) & (ntaus == 0))

        onemuon = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        # muonkin = ((leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        # muonDphiAK8 = (abs(leadingmuon.delta_phi(events.FatJet)) > 2*np.pi/3)

        region = nolepton ## Use this option to let more data through the cuts
        # region = onemuon ## Use this option to let less data through the cuts


        events['FatJet', 'btag_count'] = ak.sum(events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)].btagDeepFlavB > 0.3040, axis=1)
        events['FatJet', 'trigger_mask'] = trigger

        if ('hgg' in dataset) or ('hbb' in dataset) or ('flat' in dataset) or ('hww' in dataset):
            print(f'Higgs {dataset}')
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, 25)
            do_li = True
        elif ('wqq' in dataset) or ('ww' in dataset) or ('wlnu' in dataset) and ('hww' not in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, 24)
            do_li = True
        elif ('zqq' in dataset) or ('zz' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, 23)
            do_li = True
        elif ('wz' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, is_wz=True)
            do_li = True
        else:
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon)
            do_li = False

        events["goodjets"] = events.FatJet[fatjetSelect]
        mask = ~ak.is_none(ak.firsts(events.goodjets))
        events = events[mask]
        
        if do_li:
            events['goodjets'] = events.goodjets[(ak.local_index(events.goodjets, axis=1) == 0)]
        
        skim = ak.zip(
            {
                'goodjets':events.goodjets,
                'event': events.event,
                'GenJetAK8': events.GenJetAK8,
                'GenPart': events.GenPart,
            },
            depth_limit=1,
        )

        path = f"/project01/ndcms/cmoore24/skims/gluon_finding2/mc/samples/{dataset}"
        skim_task = dak.to_parquet(
            skim,
            path, ##Change this to where you'd like the output to be written
            compute=False,
        )
        return skim_task

    ###### Uncomment this regeion if you want to run only one of the subsamples found in input_datasets.json ######

    subset = {}
    batch = {}
    # to_skim = 'qcd' ## Use this string to choose the subsample. Name must be found in input_datasets.json ######
    for to_skim in samples_ready:
        # if (skim_ds in to_skim):
        if (f'{index}' in to_skim):
            subset[to_skim] = samples_ready[to_skim]
            files = subset[to_skim]['files']
            form = subset[to_skim]['form']
            dict_meta = subset[to_skim]['metadata']
            keys = list(files.keys())
        
            batch[to_skim] = {}
            batch[to_skim]['files'] = {}
            batch[to_skim]['form'] = form
            batch[to_skim]['metadata'] = dict_meta
        
            # for i in range(6, 7):
            for i in range(len(files)):
                # print(keys[i], flush=True)
                batch[to_skim]['files'][keys[i]] = files[keys[i]]
        else:
            continue
            
    # with open('to_reduce.json', 'w') as f:
    #     json.dump(batch, f)
            
    tasks = dataset_tools.apply_to_fileset(
        analysis,
        # samples_ready, ## Run over all subsamples in input_datasets.json
        batch, ## Run over only the subsample specified as the "to_skim" string
        uproot_options={"allow_read_errors_with_report": False},
        schemaclass = PFNanoAODSchema,
    )#[0]

    print('start compute')
    computed = dask.compute(
            tasks,
            scheduler=m.get,
            # scheduling_mode="breadth-first",
            worker_transfers=True,
            resources={"cores": 1},
            task_mode="function-calls",
            lib_resources={'cores': 24, 'slots': 24},
            prune_depth=2, 
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))
