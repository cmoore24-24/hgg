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
#from variable_functions import *
import scipy
#from coffea.ml_tools.torch_wrapper import torch_wrapper
import pickle
import argparse


full_start = time.time()
parser = argparse.ArgumentParser(description='Run ECF Skimmer')
parser.add_argument("to_skim", type=str, default=None, help='Name of dataset to process')
args = parser.parse_args()

if __name__ == "__main__":
    m = DaskVine(
        [9102, 9200],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    m.tune("temp-replica-count", 3)
    m.tune("transfer-temps-recovery", 1)
    
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


    ######## Uncomment the following section if you need to pre-process the datasets present in input_datasets.json ########
    
    # with open('input_datasets.json', 'r') as f:
    #     samples = json.load(f)

    # print('doing samples')
    # sample_start = time.time()

    # @dask.delayed
    # def sampler(samples):
    #     samples_ready, samples = dataset_tools.preprocess(
    #         samples,
    #         step_size=12_000, ## Change this step size to adjust the size of chunks of events
    #         skip_bad_files=True,
    #         recalculate_steps=True,
    #         save_form=False,
    #     )
    #     return samples_ready

    # sampler_dict = {}
    # for i in samples:
    #     sampler_dict[i] = sampler(samples[i])

    # print('Compute')
    # samples_postprocess = dask.compute(
    #     sampler_dict,
    #     scheduler=m.get,
    #     resources={"cores": 1},
    #     resources_mode=None,
    #     prune_files=True,
    #     lazy_transfers=True,
    #     #task_mode="function_calls",
    #     lib_resources={'cores': 12, 'slots': 12},
    # )[0]

    # samples_ready = {}
    # for i in samples_postprocess:
    #     samples_ready[i] = samples_postprocess[i]['files']

    # sample_stop = time.time()
    # print('samples done')
    # print('full sample time is ' + str((sample_stop - sample_start)/60))

    # with open("samples_ready.json", "w") as fout:
    #     json.dump(samples_ready, fout)

    ######## The analysis portion begins here ########
    
    with open("samples_ready.json", 'r') as f:
        samples_ready = json.load(f)

    with open('triggers.json', 'r') as f:
        triggers = json.load(f)
    
    def apply_selections(events, trigger, pdgid=None, is_wz=False):     
        fatjetSelect = (
            (events.FatJet.pt > 450)
            & (events.FatJet.pt < 600)
            & (abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.msoftdrop > 80)
            & (events.FatJet.msoftdrop < 170)
            #& (region)
            & (trigger)
            #& (events.btag_count == 0)
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
        
    # class EnergyCorrelatorFunctionTagger(torch_wrapper):
    #     def prepare_awkward(self, events, scaler, imap):
    #         fatjets = events
    
    #         retmap = {
    #             k: ak.concatenate([x[:, np.newaxis] for x in imap[k].values()], axis=1)
    #             for k in imap.keys()
    #         }
    #         x = ak.values_astype(scaler.transform(retmap['vars']), "float32")
    #         return (x,), {}

    # def imapper(array, ratio_list):
    #     imap = {}
    #     imap['vars'] = {}
    #     for i in ratio_list:
    #         try:
    #             imap['vars'][i] = array[i]
    #         except:
    #             imap['vars'][i] = array[i]
    #     return imap


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

        events['FatJet', 'num_fatjets'] = ak.num(events.FatJet)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
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

        # nolepton = ((nmuons == 0) & (nelectrons == 0) & (ntaus == 0))

        # onemuon = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        # muonkin = ((leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        # muonDphiAK8 = (abs(leadingmuon.delta_phi(events.FatJet)) > 2*np.pi/3)


        
        # num_sub = ak.unflatten(num_subjets(events.FatJet, cluster_val=0.4), counts=ak.num(events.FatJet))
        # events['FatJet', 'num_subjets'] = num_sub

        #region = nolepton ## Use this option to let more data through the cuts
        # region = onemuon ## Use this option to let less data through the cuts

        events['FatJet', 'nmuons'] = nmuons
        events['FatJet', 'nelectrons'] = nelectrons
        events['FatJet', 'ntaus'] = ntaus
        events['FatJet', 'btag_count'] = ak.sum(events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)].btagDeepFlavB > 0.3040, axis=1)

        if ('hgg' in dataset) or ('hbb' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, trigger, 25)
            do_li = False
        elif ('wqq' in dataset) or ('ww' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, trigger, 24)
            do_li = False
        elif ('zqq' in dataset) or ('zz' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, trigger, 23)
            do_li = False
        elif ('wz' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, trigger, is_wz=True)
            do_li = False
        else:
            print(dataset)
            fatjetSelect = apply_selections(events, trigger)
            do_li = True

        events["goodjets"] = events.FatJet[fatjetSelect]
        mask = ~ak.is_none(ak.firsts(events.goodjets))
        events = events[mask]
        
        if do_li:
            events['goodjets'] = events.goodjets[(ak.local_index(events.goodjets, axis=1) == 0)]
        
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )
        pf = ak.flatten(events.goodjets.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)

        events['goodjets', 'lund_decluster'] = ak.flatten(cluster.exclusive_jets_lund_declusterings(1))

        skim = ak.zip(
            {
                'goodjets':events.goodjets,
            },
            depth_limit=1,
        )
        
        skim_task = dak.to_parquet(
            skim,
            f"/project01/ndcms/cmoore24/skims/lund_output/{dataset}/", ##Change this to where you'd like the output to be written
            compute=False,
        )
        return skim_task

    ###### Uncomment this regeion if you want to run only one of the subsamples found in input_datasets.json ######

    subset = {}
    to_skim = args.to_skim ## Use this string to choose the subsample. Name must be found in input_datasets.json ######
    subset[to_skim] = samples_ready[to_skim]
    files = subset[to_skim]['files']
    form = subset[to_skim]['form']
    dict_meta = subset[to_skim]['metadata']
    keys = list(files.keys())

    batch = {}
    batch[to_skim] = {}
    batch[to_skim]['files'] = {}
    batch[to_skim]['form'] = form
    batch[to_skim]['metadata'] = dict_meta

    #for i in range(0, 100):
    for i in range(len(files)):
        batch[to_skim]['files'][keys[i]] = files[keys[i]]
    
    tasks = dataset_tools.apply_to_fileset(
        analysis,
        #samples_ready, ## Run over all subsamples in input_datasets.json
        batch, ## Run over only the subsample specified as the "to_skim" string
        uproot_options={"allow_read_errors_with_report": False},
        schemaclass = PFNanoAODSchema,
    )#[0]

    print('start compute')
    computed = dask.compute(
            tasks,
            scheduler=m.get,
            resources={"cores": 1},
            resources_mode=None,
            lazy_transfers=False,
            prune_files=True,
            #task_mode="function_calls",
            lib_resources={'cores': 12, 'slots': 12},
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))