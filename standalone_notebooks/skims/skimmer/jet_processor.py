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
#from coffea.ml_tools.torch_wrapper import torch_wrapper
import pickle
import subprocess


full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9101, 9200],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info/JetHTF",
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


    ####### Uncomment the following section if you need to pre-process the datasets present in input_datasets.json ########
    
    # with open('input_datasets.json', 'r') as f:
    #     samples = json.load(f)

    # print('doing samples')
    # sample_start = time.time()

    # @dask.delayed
    # def sampler(samples):
    #    samples_ready, samples = dataset_tools.preprocess(
    #        samples,
    #        step_size=1000, ## Change this step size to adjust the size of chunks of events
    #        skip_bad_files=True,
    #        recalculate_steps=True,
    #        save_form=False,
    #    )
    #    return samples_ready
    
    # sampler_dict = {}
    # for i in samples:
    #    sampler_dict[i] = sampler(samples[i])
    
    # print('Compute')
    # samples_postprocess = dask.compute(
    #    sampler_dict,
    #    scheduler=m.get,
    #    resources={"cores": 8},
    #    resources_mode=None,
    #    prune_files=True,
    #    lazy_transfers=True,
    #    #task_mode="function_calls",
    #    lib_resources={'cores': 12, 'slots': 12},
    # )[0]
    
    # samples_ready = {}
    # for i in samples_postprocess:
    #    samples_ready[i] = samples_postprocess[i]['files']
    
    # sample_stop = time.time()
    # print('samples done')
    # print('full sample time is ' + str((sample_stop - sample_start)/60))
    
    # with open("samples_ready.json", "w") as fout:
    #    json.dump(samples_ready, fout)

    ######## The analysis portion begins here ########
    
    with open("samples_ready.json", 'r') as f:
        samples_ready = json.load(f)

    with open('triggers.json', 'r') as f:
        triggers = json.load(f)

    # ecf_list = []
    # for ratio in ratio_list:
    #     dash = ratio.find('/')
    #     asterisk = ratio.find('*')
    #     numerator = ratio[:dash]
    #     denominator = ratio[dash+1:asterisk]
    #     ecf_list.append(numerator)
    #     ecf_list.append(denominator)
    # ecf_list = list(set(ecf_list))

    def apply_selections(events, region, trigger, pdgid=None, is_wz=False):     
        fatjetSelect = (
            (events.FatJet.pt >= 450)
            & (events.FatJet.pt <= 1200)
            & (abs(events.FatJet.eta) <= 2.4)
            & (events.FatJet.msoftdrop >= 40)
            & (events.FatJet.msoftdrop <= 200)
            & (region)
            # & (trigger)
            # & (events.FatJet.btag_count == 0)
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
        
    def ecf_reorg(ecf_dict, jet_array):
        output_dict = {}        
        for i in ecf_dict:
            if i[1] == '2':
                output_dict[f'1{i}'] = ak.unflatten(ecf_dict[i], counts = ak.num(jet_array))
            elif i[1] == '3':
                output_dict[f'1{i}'] = ak.unflatten(ecf_dict[i][:,0], counts = ak.num(jet_array))
                output_dict[f'2{i}'] = ak.unflatten(ecf_dict[i][:,1], counts = ak.num(jet_array))
                output_dict[f'3{i}'] = ak.unflatten(ecf_dict[i][:,2], counts = ak.num(jet_array))
            elif i[1] == '4':
                output_dict[f'1{i}'] = ak.unflatten(ecf_dict[i][:,0], counts = ak.num(jet_array))
                output_dict[f'2{i}'] = ak.unflatten(ecf_dict[i][:,1], counts = ak.num(jet_array))
                output_dict[f'3{i}'] = ak.unflatten(ecf_dict[i][:,2], counts = ak.num(jet_array))
                output_dict[f'4{i}'] = ak.unflatten(ecf_dict[i][:,3], counts = ak.num(jet_array))
                output_dict[f'5{i}'] = ak.unflatten(ecf_dict[i][:,4], counts = ak.num(jet_array))
                output_dict[f'6{i}'] = ak.unflatten(ecf_dict[i][:,5], counts = ak.num(jet_array))
            elif i[1] == '5':
                output_dict[f'1{i}'] = ak.unflatten(ecf_dict[i][:,0], counts = ak.num(jet_array))
                output_dict[f'2{i}'] = ak.unflatten(ecf_dict[i][:,1], counts = ak.num(jet_array))
                output_dict[f'3{i}'] = ak.unflatten(ecf_dict[i][:,2], counts = ak.num(jet_array))
                output_dict[f'4{i}'] = ak.unflatten(ecf_dict[i][:,3], counts = ak.num(jet_array))
                output_dict[f'5{i}'] = ak.unflatten(ecf_dict[i][:,4], counts = ak.num(jet_array))
                output_dict[f'6{i}'] = ak.unflatten(ecf_dict[i][:,5], counts = ak.num(jet_array))
                output_dict[f'7{i}'] = ak.unflatten(ecf_dict[i][:,6], counts = ak.num(jet_array))
                output_dict[f'8{i}'] = ak.unflatten(ecf_dict[i][:,7], counts = ak.num(jet_array))
                output_dict[f'9{i}'] = ak.unflatten(ecf_dict[i][:,8], counts = ak.num(jet_array))
                output_dict[f'10{i}'] = ak.unflatten(ecf_dict[i][:,9], counts = ak.num(jet_array))
        return output_dict
    

    def labels(events, to_be_true):
            events['goodjets', 'label_H_gg'] = ak.zeros_like(events.goodjets.pt)
            events['goodjets', 'label_QCD'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_H_bb'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_Wqq'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_Zqq'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_WW'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_WZ'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_ZZ'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_TTBoosted'] = ak.zeros_like(events.goodjets.pt)
            # events['goodjets', 'label_Singletop'] = ak.zeros_like(events.goodjets.pt)
            if to_be_true != None:
                events['goodjets', to_be_true] = ak.ones_like(events.goodjets.pt)
            return events

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

        nolepton = ((nmuons == 0) & (nelectrons == 0) & (ntaus == 0))

        onemuon = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        # muonkin = ((leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        # muonDphiAK8 = (abs(leadingmuon.delta_phi(events.FatJet)) > 2*np.pi/3)


        
        # num_sub = ak.unflatten(num_subjets(events.FatJet, cluster_val=0.4), counts=ak.num(events.FatJet))
        # events['FatJet', 'num_subjets'] = num_sub

        region = nolepton ## Use this option to let more data through the cuts
        # region = onemuon ## Use this option to let less data through the cuts


        events['FatJet', 'btag_count'] = ak.sum(events.Jet[(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)].btagDeepFlavB > 0.3040, axis=1)
        events['FatJet', 'trigger_mask'] = trigger

        if ('hgg' in dataset) or ('hbb' in dataset) or ('flat' in dataset):
            print(f'Higgs {dataset}')
            fatjetSelect = apply_selections(events, region, trigger, 25)
            do_li = False
        elif ('wqq' in dataset) or ('ww' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, 24)
            do_li = False
        elif ('zqq' in dataset) or ('zz' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, 23)
            do_li = False
        elif ('wz' in dataset):
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger, is_wz=True)
            do_li = False
        else:
            print(dataset)
            fatjetSelect = apply_selections(events, region, trigger)
            do_li = True

        events["goodjets"] = events.FatJet[fatjetSelect]
        mask = ~ak.is_none(ak.firsts(events.goodjets))
        events = events[mask]
        
        if do_li:
            events['goodjets'] = events.goodjets[(ak.local_index(events.goodjets, axis=1) == 0)]
        
        events['goodjets', 'color_ring'] = ak.unflatten(
             color_ring(events.goodjets, cluster_val=0.4), counts=ak.num(events.goodjets)
        )

        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 1.0
        )
        pf = ak.flatten(events.goodjets.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)

        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)

        #### all angles, if possible 
        
        ungroomed_ecf_classes = {}
        for n in range(2, 6):
            for b in range(5, 45, 5):
                ecf_class = f'e{n}^{b/10}'
                ecf_result = cluster.exclusive_jets_energy_correlator(
                        func='generalized', npoint=n, beta=b/10, normalized=True, all_angles=True
                )
                ungroomed_ecf_classes[ecf_class] = ak.unflatten(ecf_result, counts = ((n*(n-1))/2))
                
        groomed_ecf_classes = {}
        for n in range(2, 6):
            for b in range(5, 45, 5):
                ecf_class = f'e{n}^{b/10}'
                ecf_result = softdrop_cluster.exclusive_jets_energy_correlator(
                            func='generalized', npoint=n, beta=b/10, normalized=True, all_angles=True
                )
                groomed_ecf_classes[ecf_class] = ak.unflatten(ecf_result, counts = ((n*(n-1))/2))
                
        ungroomed_ecfs = ecf_reorg(ungroomed_ecf_classes, events.goodjets)
        groomed_ecfs = ecf_reorg(groomed_ecf_classes, events.goodjets)
        events["groomed_ecfs"] = ak.zip(groomed_ecfs, depth_limit=1)
        events["ungroomed_ecfs"] = ak.zip(ungroomed_ecfs, depth_limit=1)

        #### slower ecfs

        # groomed_ecfs = {}
        # for n in range(2,6):
        #     for v in range(1, int(scipy.special.binom(n, 2))+1):
        #         for b in range(5, 45, 5):
        #             ecf_name = f'{v}e{n}^{b/10}'
        #             groomed_ecfs[ecf_name] = ak.unflatten(
        #                 softdrop_cluster.exclusive_jets_energy_correlator(
        #                     func='generalized', npoint=n, angles=v, beta=b/10, normalized=True), 
        #                 counts=ak.num(events.goodjets)
        #             )

        # ungroomed_ecfs = {}
        # for n in range(2,6):
        #     for v in range(1, int(scipy.special.binom(n, 2))+1):
        #         for b in range(5, 45, 5):
        #             ecf_name = f'{v}e{n}^{b/10}'
        #             ungroomed_ecfs[ecf_name] = ak.unflatten(
        #                 cluster.exclusive_jets_energy_correlator(
        #                     func='generalized', npoint=n, angles=v, beta=b/10, normalized=True), 
        #                 counts=ak.num(events.goodjets)
        #             )
        
        # events["groomed_ecfs"] = ak.zip(groomed_ecfs, depth_limit=1)
        # events["ungroomed_ecfs"] = ak.zip(ungroomed_ecfs, depth_limit=1)

        
        if ('hgg' in dataset) or ('flat' in dataset):
            events = labels(events, 'label_H_gg') 
        elif ('qcd' in dataset):
            events = labels(events, 'label_QCD') 
        else: #('wqq' in dataset):
            events = labels(events, None) 
        # elif ('zqq' in dataset):
        #     events = labels(events, 'label_Zqq') 
        # elif ('ww' in dataset):
        #     events = labels(events, 'label_WW')
        # elif ('wz' in dataset):
        #     events = labels(events, 'label_WZ')
        # elif ('zz' in dataset):
        #     events = labels(events, 'label_ZZ')
        # elif ('ttboosted' in dataset):
        #     events = labels(events, 'label_TTBoosted')
        # elif ('singletop' in dataset):
        #     events = labels(events, 'label_Singletop')
        # elif ('hbb' in dataset):
        #     events = labels(events, 'label_H_bb')

        pfcands = events.goodjets.constituents.pf
        goodjets = ak.flatten(events.goodjets)
        sv = events.SV

        pn_vals = ak.zip(
            {
                'pfcand_eta':ak.flatten(pfcands.eta),
                'pfcand_phi':ak.flatten(pfcands.phi),
                'pfcand_charge':ak.flatten(pfcands.charge),
                'pfcand_d0':ak.flatten(pfcands.d0),
                'pfcand_dz':ak.flatten(pfcands.dz),
                'pfcand_lostInnerHits':ak.flatten(pfcands.lostInnerHits),
                'pfcand_pt':ak.flatten(pfcands.pt),
                # 'pfcand_eta':pfcands.eta,
                # 'pfcand_phi':pfcands.phi,
                # 'pfcand_charge':pfcands.charge,
                # 'pfcand_d0':pfcands.d0,
                # 'pfcand_dz':pfcands.dz,
                # 'pfcand_lostInnerHits':pfcands.lostInnerHits,
                # 'pfcand_pt':pfcands.pt,
                'sv_eta':sv.eta,
                'sv_phi':sv.phi,
                'sv_dxy':sv.dxy,
                'sv_mass':sv.mass,
                'sv_chi2':sv.chi2,
                'sv_ntracks':sv.ntracks,
                'sv_pt':sv.pt,
                'fj_sdmass':goodjets.msoftdrop,
                'fj_pt':goodjets.pt,
                'fj_eta':goodjets.eta,
                'fj_phi':goodjets.phi,
                'label_H_gg':goodjets.label_H_gg,
                'label_QCD':goodjets.label_QCD,             
            },
            depth_limit=1,
        )

        skim = ak.zip(
            {
                'goodjets':ak.firsts(events.goodjets),
                'ungroomed_ecfs':ak.firsts(events.ungroomed_ecfs),
                'groomed_ecfs':ak.firsts(events.groomed_ecfs),
                'event': events.event,
                'pnet_vals': pn_vals,
            },
            depth_limit=1,
        )

        path = f"/project01/ndcms/cmoore24/skims/full_skims/nolepton/data/{dataset}"
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
        if ('F' in to_skim):
            subset[to_skim] = samples_ready[to_skim]
            files = subset[to_skim]['files']
            form = subset[to_skim]['form']
            dict_meta = subset[to_skim]['metadata']
            keys = list(files.keys())
        
            batch[to_skim] = {}
            batch[to_skim]['files'] = {}
            batch[to_skim]['form'] = form
            batch[to_skim]['metadata'] = dict_meta
        
            #for i in range(0, 100):
            for i in range(len(files)):
                batch[to_skim]['files'][keys[i]] = files[keys[i]]
        else:
            continue
            
    with open('to_reduce.json', 'w') as f:
        json.dump(batch, f)
            
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
            scheduling_mode="depth-first",
            resources={"cores": 1},
            resources_mode=None,
            lazy_transfers=False,
            prune_files=True,
            #task_mode="function-calls",
            #lib_resources={'cores': 12, 'slots': 12},
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))
