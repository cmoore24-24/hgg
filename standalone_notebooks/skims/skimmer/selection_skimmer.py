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

full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
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
    
    # with open('input_datasets.json', 'r') as f:
    #     samples = json.load(f)

    # print('doing samples')
    # sample_start = time.time()

    # @dask.delayed
    # def sampler(samples):
    #     samples_ready, samples = dataset_tools.preprocess(
    #         samples,
    #         step_size=45_000,
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

    with open("samples_ready.json", 'r') as fin:
        samples_ready = json.load(fin)

    with open('triggers.json', 'r') as f:
        triggers = json.load(f)

    def analysis(events):
        dataset = events.metadata["dataset"]
        print(dataset)

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

        nolepton = ((nmuons == 0) & (nelectrons == 0) & (ntaus == 0))

        # onemuon = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        # muonkin = ((leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        # muonDphiAK8 = (abs(leadingmuon.delta_phi(events.FatJet)) > 2*np.pi/3)

        # num_sub = ak.unflatten(num_subjets(events.FatJet, cluster_val=0.4), counts=ak.num(events.FatJet))
        # events['FatJet', 'num_subjets'] = num_sub


        events['btag_count'] = ak.sum(events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)].btagDeepFlavB > 0.3040, axis=1)

        if ('hgg' in dataset) or ('hbb' in dataset):
            print("signal")
            genhiggs = events.GenPart[
                (events.GenPart.pdgId == 25)
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
            parents = events.FatJet.nearest(genhiggs, threshold=0.2)
            higgs_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 1200)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (nolepton)
                & (trigger)
                & (higgs_jets)
            )

        elif ('wqq' in dataset) or ('ww' in dataset):
            print('w background')
            genw = events.GenPart[
                (abs(events.GenPart.pdgId) == 24)
                & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
            ]
            parents = events.FatJet.nearest(genw, threshold=0.2)
            w_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 1200)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (trigger)
                & (nolepton)
                & (w_jets)
            )

        elif ('zqq' in dataset) or ('zz' in dataset):
            print('z background')
            genz = events.GenPart[
                (events.GenPart.pdgId == 23)
                & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
            ]
            parents = events.FatJet.nearest(genz, threshold=0.2)
            z_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 1200)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (nolepton)
                & (trigger)
                & (z_jets)
            )

        elif ('wz' in dataset):
            print('wz background')
            genwz = events.GenPart[
                ((abs(events.GenPart.pdgId) == 24)|(events.GenPart.pdgId == 23))
                & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
            ]
            parents = events.FatJet.nearest(genwz, threshold=0.2)
            wz_jets = ~ak.is_none(parents, axis=1)

            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 1200)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (nolepton)
                & (trigger)
                & (wz_jets)
            )
    
        else:
            print('background')
            fatjetSelect = (
                (events.FatJet.pt > 400)
                #& (events.FatJet.pt < 1200)
                & (abs(events.FatJet.eta) < 2.4)
                & (events.FatJet.msoftdrop > 40)
                & (events.FatJet.msoftdrop < 200)
                & (trigger)
                & (nolepton)
            )
        
        events["goodjets"] = events.FatJet[fatjetSelect]
        mask = ~ak.is_none(ak.firsts(events.goodjets))
        events = events[mask]
        
        events['goodjets', 'color_ring'] = ak.unflatten(
             color_ring(events.goodjets, cluster_val=0.4), counts=ak.num(events.goodjets)
        )


        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )
        pf = ak.flatten(events.goodjets.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        
        # ecf_dict = {}

        # ecf_dict['1e2^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=2, beta=0.5
        #     ) 

        # ecf_dict['1e2^1.0'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=2, beta=1.0
        #     ) 

        # ecf_dict['1e2^1.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=2, beta=1.5
        #     )

        # ecf_dict['1e2^2.0'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=2, beta=2.0
        #     )

        # ecf_dict['1e2^2.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=2, beta=2.5
        #     )

        # ecf_dict['1e3^1.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=3, beta=1.5
        #     )

        # ecf_dict['1e3^2.0'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=3, beta=2.0
        #     )

        # ecf_dict['2e3^1.0'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=2, npoint=3, beta=1.0
        #     )

        # ecf_dict['2e3^1.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=2, npoint=3, beta=1.5
        #     )

        # ecf_dict['2e3^2.0'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=2, npoint=3, beta=2.0
        #     )

        # ecf_dict['3e3^1.0'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=3, npoint=3, beta=1.0
        #     )
        
        # ecf_dict['1e4^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=4, beta=0.5
        #     )

        # ecf_dict['3e4^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=3, npoint=4, beta=0.5
        #     )

        # ecf_dict['3e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=3, npoint=5, beta=0.5
        #     )

        # ecf_dict['1e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=1, npoint=5, beta=0.5
        #     )
        
        # ecf_dict['2e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=2, npoint=5, beta=0.5
        #     )

        # ecf_dict['4e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=4, npoint=5, beta=0.5
        #     )

        # ecf_dict['5e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=5, npoint=5, beta=0.5
        #     )

        # ecf_dict['7e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=7, npoint=5, beta=0.5
        #     )

        # ecf_dict['8e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=8, npoint=5, beta=0.5
        #     )

        # ecf_dict['9e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=9, npoint=5, beta=0.5
        #     )

        # ecf_dict['10e5^0.5'] = softdrop_cluster.exclusive_jets_energy_correlator(
        #         func='generic', angles=10, npoint=5, beta=0.5
        #     )


        # events['goodjets', '1e4^0.5/1e2^0.5**1.0'] = ak.unflatten(
        #     ((ecf_dict['1e4^0.5']/ecf_dict['1e2^0.5'])**1.0), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '1e4^0.5/1e2^1.0**0.5'] = ak.unflatten(
        #     ((ecf_dict['1e4^0.5']/ecf_dict['1e2^1.0'])**0.5), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '2e5^0.5/1e2^0.5**2.0'] = ak.unflatten(
        #     ((ecf_dict['2e5^0.5']/ecf_dict['1e2^0.5'])**2.0), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '4e5^0.5/1e3^1.5**1.3333333333333333'] = ak.unflatten(
        #     ((ecf_dict['4e5^0.5']/ecf_dict['1e3^1.5'])**(2/1.5)), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '8e5^0.5/2e3^1.5**1.3333333333333333'] = ak.unflatten(
        #     ((ecf_dict['8e5^0.5']/ecf_dict['2e3^1.5'])**(4/3)), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '10e5^0.5/2e3^2.0**1.25'] = ak.unflatten(
        #     ((ecf_dict['10e5^0.5']/ecf_dict['2e3^2.0'])**1.25), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '1e4^0.5/1e2^1.5**0.3333333333333333'] = ak.unflatten(
        #     ((ecf_dict['1e4^0.5']/ecf_dict['1e2^1.5'])**(.5/1.5)), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '1e4^0.5/1e2^2.0**0.25'] = ak.unflatten(
        #     ((ecf_dict['1e4^0.5']/ecf_dict['1e2^2.0'])**0.25), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '1e4^0.5/1e2^2.5**0.2'] = ak.unflatten(
        #     ((ecf_dict['1e4^0.5']/ecf_dict['1e2^2.5'])**0.2), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '1e5^0.5/1e2^2.0**0.25'] = ak.unflatten(
        #     ((ecf_dict['1e5^0.5']/ecf_dict['1e2^2.0'])**0.25), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '2e5^0.5/3e4^0.5**0.6666666666666666'] = ak.unflatten(
        #     ((ecf_dict['2e5^0.5']/ecf_dict['3e4^0.5'])**(1/1.5)), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '3e5^0.5/1e3^1.5**1.0'] = ak.unflatten(
        #     ((ecf_dict['3e5^0.5']/ecf_dict['1e3^1.5'])**1.0), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '4e5^0.5/1e3^2.0**1.0'] = ak.unflatten(
        #     ((ecf_dict['4e5^0.5']/ecf_dict['1e3^2.0'])**1.0), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '5e5^0.5/1e3^2.0**1.25'] = ak.unflatten(
        #     ((ecf_dict['5e5^0.5']/ecf_dict['1e3^2.0'])**1.25), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '5e5^0.5/2e3^1.0**1.25'] = ak.unflatten(
        #     ((ecf_dict['5e5^0.5']/ecf_dict['2e3^1.0'])**1.25), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '7e5^0.5/2e3^1.5**1.1666666666666667'] = ak.unflatten(
        #     ((ecf_dict['7e5^0.5']/ecf_dict['2e3^1.5'])**(3.5/3)), 
        #     counts=ak.num(events.goodjets)
        # )

        # events['goodjets', '9e5^0.5/3e3^1.0**1.5'] = ak.unflatten(
        #     ((ecf_dict['9e5^0.5']/ecf_dict['3e3^1.0'])**1.5), 
        #     counts=ak.num(events.goodjets)
        # )

        skim_task = dak.to_parquet(
            events,
            f"/project01/ndcms/cmoore24/skims/ecfs/new_ecf_skims/{dataset}",
            compute=False,
        )
        return skim_task


    subset = {}
    batch = {}
    
    # to_skim = 'qcd_3200toInf'
    # subset[to_skim] = samples_ready[to_skim]

    to_skim = []
    for i in samples_ready:
        #if 'qcd' in i: #QCD Samples
        #if 'w' in i: #W samples
        #if ('z' in i) and ('wz' not in i): #Z samples
        #if ('qcd' not in i) and ('w' not in i) and ('z' not in i): #Everything Else
            to_skim.append(i)
    for i in to_skim:
        subset[i] = samples_ready[i]

    for i in subset:
        files = subset[i]['files']
        form = subset[i]['form']
        dict_meta = subset[i]['metadata']
        keys = list(files.keys())
    

        batch[i] = {}
        batch[i]['files'] = {}
        batch[i]['form'] = form
        batch[i]['metadata'] = dict_meta
    
        for j in range(len(files)):
            batch[i]['files'][keys[j]] = files[keys[j]]
    
    tasks = dataset_tools.apply_to_fileset(
        analysis,
        #dataset_tools.slice_files(samples_ready, slice(None, 20)),
        #dataset_tools.slice_files(batch, slice(None, 100, None)),
        #samples_ready,
        batch,
        uproot_options={"allow_read_errors_with_report": False},
        schemaclass = PFNanoAODSchema,
    )


    print('start compute')
    computed = dask.compute(
            tasks,
            scheduler=m.get,
            resources={"cores": 1},
            resources_mode=None,
            lazy_transfers=True,
            prune_files=True,
            #task_mode="function_calls",
            lib_resources={'cores': 12, 'slots': 12},
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))
