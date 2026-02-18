import json
import dask
import dask_awkward as dak
import awkward as ak
import numpy as np
from coffea import dataset_tools
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema, NanoAODSchema
from ndcctools.taskvine import DaskVine
import fastjet
import time
import os
import warnings
import scipy
import pickle
from coffea.lumi_tools import LumiMask

index = 'C'
samples_process = True
lep_region = 'trijet'
template_name = f'{lep_region}_{index}'
dset_type = 'data'
year = '2017'
match = True
full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9101, 9200],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info/",
        run_info_template=f'{template_name}',
    )

    m.tune("temp-replica-count", 3)
    m.tune("worker-source-max-transfers", 1000)
    m.tune("immediate-recovery", 0)
    m.tune("max-retrievals", 1)
    m.tune("transient-error-interval", 1)
    m.tune("prefer-dispatch", 1)
    
    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "invalid value")

    ##### Turns input datasets into post-processed, stepped filelist #####

    if samples_process:
        with open('input_datasets.json', 'r') as f:
            samples = json.load(f)
    
        print('doing samples')
        sample_start = time.time()
    
        @dask.delayed
        def sampler(samples):
           samples_ready, samples = dataset_tools.preprocess(
               samples,
               step_size=7500, ## Change this step size to adjust the size of chunks of events
               skip_bad_files=False,
               recalculate_steps=True,
               save_form=True,
           )
           return samples_ready
        
        sampler_dict = {}
        for i in samples:
           sampler_dict[i] = sampler(samples[i])
        
        print('Compute')
        samples_postprocess = dask.compute(
           sampler_dict,
           scheduler=m.get,
           resources={"cores": 2},
           # resources_mode=None,
           # prune_files=True,
           prune_depth=0,
           worker_transfers=True,
           task_mode="function-calls",
           lib_resources={'cores': 2, 'slots': 1},
        )[0]
        
        samples_ready = {}
        with open('test_file.json','w') as f:
            json.dump(samples_postprocess,f)
            
        for i in samples_postprocess:
           samples_ready[i] = samples_postprocess[i]['files']
        
        sample_stop = time.time()
        print('samples done')
        print('full sample time is ' + str((sample_stop - sample_start)/60))

        
        with open("samples_ready.json", "w") as fout:
           json.dump(samples_ready, fout)

    ##### Begin Analysis #####

    ##### Function Definitions #####

    with open("samples_ready.json", 'r') as f:
        samples_ready = json.load(f)

    print(f'The region is {lep_region}, the year is {year}, gen matching is {match}, and the section is {dset_type}.')
        
    if (lep_region == 'nolepton') or (lep_region == 'trijet'):
        with open('triggers.json', 'r') as f:
            triggers = json.load(f)
    elif lep_region == 'singlemuon':
        triggers = {}
        triggers['2017'] = ['Mu50','TkMu50']
        triggers['2016'] = ['Mu50']
        triggers['2018'] = ['Mu50']

    def apply_selections(events, region, trigger, goodmuon, do_matching=True, pdgid=None, is_wz=False):     
        fatjetSelect = (
            # (events.FatJet.pt >= 450) &
            (events.FatJet.pt <= 1200)
            & (abs(events.FatJet.eta) <= 2.4)
            & (events.FatJet.msoftdrop >= 40)
            & (events.FatJet.msoftdrop <= 200)
            & (region)
            & (trigger)
            & (events.FatJet.btag_count == 0)
        )

        if lep_region == 'nolepton':
            fatjetSelect = ((fatjetSelect) 
                            & (ak.num(events.FatJet) < 3) 
                            & (events.FatJet.pt >= 450))
        elif lep_region == 'singlemuon':
            fatjetSelect = ((fatjetSelect) & (ak.num(events.FatJet) < 3) & (events.FatJet.pt >= 400) & (ak.fill_none(events.FatJet.delta_r(events.FatJet.nearest(events.Muon[goodmuon], axis=1)) > 0.8, True)))
        elif lep_region == 'trijet':
            fatjetSelect = ((fatjetSelect) & (ak.num(events.FatJet) >= 3) & (events.FatJet.pt >= 450))

        if do_matching:
            if ((pdgid != None) or (is_wz)):
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
                fatjetSelect = (fatjetSelect) & (matched_jets)
                return fatjetSelect
        return fatjetSelect

    def matching_mask(events, pdgid=None, is_wz=False):
        if ((pdgid != None) or (is_wz)):
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
            return matched_jets

    def ecf_reorg(ecf_dict, jet_array):
        output_dict = {}        
        for i in ecf_dict:
            if i[1] == '2':
                # output_dict[f'1{i}'] = ak.unflatten(ecf_dict[i], counts = ak.num(jet_array))
                output_dict[f'1{i}'] = ecf_dict[i]
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
        if to_be_true != None:
            events['goodjets', to_be_true] = ak.ones_like(events.goodjets.pt)
        return events

    ##### Dataset Processor #####

    def analysis(events):
        dataset = events.metadata["dataset"]
        
        # Puppi Weighting
        events['PFCands', 'pt'] = (
            events.PFCands.pt
            * events.PFCands.puppiWeight
        )

        # Addresses some events that have no PF candidates
        cut_to_fix_softdrop = (ak.num(events.FatJet.constituents.pf, axis=2) > 0)
        events = events[ak.all(cut_to_fix_softdrop, axis=1)]

        # Create Trigger Mask
        trigger = ak.zeros_like(ak.firsts(events.FatJet.pt), dtype='bool')
        # if 'APV' not in year:
        #     trigg_year = year
        # else:
        #     trigg_year = '2016'
        for t in triggers[year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        trigger = ak.fill_none(trigger, False)
        events['FatJet', 'trigger_mask'] = trigger

        lumimasks = {
                '2016': 'lumimasks/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
                '2017': 'lumimasks/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
                '2018': 'lumimasks/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
            }

        # Store important values; number of fatjets and number of b-tagged AK4 jets per event
        events['FatJet', 'num_fatjets'] = ak.num(events.FatJet)
        events['FatJet', 'dazsle_msd'] = (events.FatJet.subjets * (1 - events.FatJet.subjets.rawFactor)).sum()

        btag_wps = {
            '2016APV': 0.3142,
            '2016': 0.3657,
            '2017': 0.3040,
            '2018': 0.2561,
        }
        events['FatJet', 'btag_count'] = ak.sum(events.Jet[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.5)].btagDeepFlavB > btag_wps[year], axis=1)

        # Create muon selections
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

        # Create masks for the 0-lepton region and the single muon region
        nolepton = ((nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        singlemuon = ((nmuons == 1) & (nelectrons == 0) & (ntaus == 0))

        if (lep_region == 'nolepton') or (lep_region == 'trijet'):
            region = nolepton
        elif lep_region == 'singlemuon':
            region = singlemuon
            
        # Apply different matching criteria depending on the input dataset
        if ('hgg' in dataset) or ('hbb' in dataset) or ('flat' in dataset) or ('hw' in dataset) or ('vbf' in dataset) or ('hz' in dataset):
            print(f'Higgs {dataset}')
            pdgId = 25
            events['FatJet','match_mask'] = matching_mask(events, pdgId)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, do_matching=match, pdgid=pdgId)
            do_li = True
        elif (('wqq' in dataset) or ('ww' in dataset) or ('wlnu' in dataset)) and ('hww' not in dataset):
            print(dataset)
            pdgId = 24
            events['FatJet','match_mask'] = matching_mask(events, pdgId)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, do_matching=match, pdgid=pdgId)
            do_li = True
        elif ('zqq' in dataset) or ('zz' in dataset):
            print(dataset)
            pdgId = 23
            events['FatJet','match_mask'] = matching_mask(events, pdgId)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, do_matching=match, pdgid=pdgId)
            do_li = True
        elif ('wz' in dataset):
            print(dataset)
            pdgId = 'wz'
            events['FatJet','match_mask'] = matching_mask(events, pdgId, is_wz=True)
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, do_matching=match, pdgid=pdgId, is_wz=True)
            do_li = True
        else:
            print(dataset)
            pdgId = None
            fatjetSelect = apply_selections(events, region, trigger, goodmuon, do_matching=match, pdgid=pdgId)
            do_li = True

        if dset_type == 'data':
            if 'APV' not in year:
                lumi_year = year
            else:
                lumi_year = '2016'
            lumimask = LumiMask(lumimasks[lumi_year])(events.run, events.luminosityBlock)
            fatjetSelect = (fatjetSelect) & (lumimask)
        
        
        # Create array with passing AK8 jets
        events["goodjets"] = events.FatJet[fatjetSelect]
        events['FatJet','goodjets'] = fatjetSelect
        mask = ~ak.is_none(ak.firsts(events.goodjets))
        events = events[mask]

        # Guarantee array only contains leading entry
        if do_li:
            events['goodjets'] = events.goodjets[(ak.local_index(events.goodjets, axis=1) == 0)]

        # Setup FastJet
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 1.0
        )
        pf = ak.flatten(events.goodjets.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)

        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)

        with open('/scratch365/cmoore24/training/hgg/final_draft_dev/QCDgg/results/individual_ecfs/groomed_ecfs.pkl', 'rb') as f:
            groomed_ecf_list = pickle.load(f)
        with open('/scratch365/cmoore24/training/hgg/final_draft_dev/QCDgg/results/individual_ecfs/ungroomed_ecfs.pkl', 'rb') as f:
            ungroomed_ecf_list = pickle.load(f)

        # Calculate energy correlation functions, requires all_angles functionality of fastjet 
        ungroomed_ecf_classes = {}
        for n in range(2, 6):
            for b in range(5, 45, 5):
                if b == 5:
                    ecf_class = f'e{n}0{b}'
                else:
                    ecf_class = f'e{n}{b}'
                ecf_result = cluster.exclusive_jets_energy_correlator(
                        func='generalized', npoint=n, beta=b/10, normalized=True, all_angles=True
                )
                ungroomed_ecf_classes[ecf_class] = ak.unflatten(ecf_result, counts = int((n*(n-1))/2))
                
        groomed_ecf_classes = {}
        for n in range(2, 6):
            for b in range(5, 45, 5):
                if b == 5:
                    ecf_class = f'e{n}0{b}'
                else:
                    ecf_class = f'e{n}{b}'
                ecf_result = softdrop_cluster.exclusive_jets_energy_correlator(
                            func='generalized', npoint=n, beta=b/10, normalized=True, all_angles=True
                )
                groomed_ecf_classes[ecf_class] = ak.unflatten(ecf_result, counts = int((n*(n-1))/2))
                
        ungroomed_ecfs_temp = ecf_reorg(ungroomed_ecf_classes, events.goodjets)
        groomed_ecfs_temp = ecf_reorg(groomed_ecf_classes, events.goodjets)
        
        groomed_ecfs = {}
        ungroomed_ecfs = {}
        
        for i in groomed_ecf_list:
            groomed_ecfs[i] = groomed_ecfs_temp[i]
        for i in ungroomed_ecf_list:
            ungroomed_ecfs[i] = ungroomed_ecfs_temp[i]

        del(ungroomed_ecfs_temp)
        del(groomed_ecfs_temp)
        
        events["groomed_ecfs"] = ak.firsts(ak.zip(groomed_ecfs, depth_limit=1))
        events["ungroomed_ecfs"] = ak.firsts(ak.zip(ungroomed_ecfs, depth_limit=1))

        #Trijet gluon variables
        
        sss_R = events.FatJet[:,1].deltaR(events.FatJet[:,2])
        num_high = ak.num(events.FatJet.nearest(events.Jet)[:,:3][events.FatJet.nearest(events.Jet)[:,:3].qgl > 0.5])
        l_msd_ratio = events.FatJet[:,0].msoftdrop / events.FatJet[:,0].mass
        ls_m_inv   = (events.FatJet[:,0] + events.FatJet[:,0]).mass
        l_qgl = events.FatJet.nearest(events.Jet)[:,0].qgl
        s_qgl = events.FatJet.nearest(events.Jet)[:,1].qgl
        ss_qgl = events.FatJet.nearest(events.Jet)[:,2].qgl
        num_high = ak.num(events.FatJet.nearest(events.Jet)[:,:3][events.FatJet.nearest(events.Jet)[:,:3].qgl > 0.5])
        
        events['extra_vars'] = ak.zip({
            'sss_R': sss_R,
            'nSVs': ak.num(events.SV),
            'l_msd_ratio': l_msd_ratio,
            'ls_m_inv': ls_m_inv,
            'l_qgl': l_qgl,
            's_qgl': s_qgl,
            'ss_qgl': ss_qgl,
            'num_high': num_high,
        },
            depth_limit=1,
        )

        # JEC Variables
        def add_jec_variables(jets, event_rho):
            jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
            jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
            if dset_type == 'mc':
                jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets["rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
            return jets

        events['goodjets','pt_raw'] = (1 - events.goodjets.rawFactor)*events.goodjets.pt
        events['goodjets','mass_raw'] = (1 - events.goodjets.rawFactor)*events.goodjets.mass
        events['goodjets','rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, events.goodjets.pt)[0]

        events['Jet','pt_raw'] = (1 - events.Jet.rawFactor)*events.Jet.pt
        events['Jet','mass_raw'] = (1 - events.Jet.rawFactor)*events.Jet.mass
        events['Jet','rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, events.Jet.pt)[0]

        if dset_type == 'mc':
            events['goodjets','pt_gen'] = ak.values_astype(ak.fill_none(events.goodjets.matched_gen.pt, 0), np.float32)
            events['Jet','pt_gen'] = ak.values_astype(ak.fill_none(events.Jet.matched_gen.pt, 0), np.float32)
        
        events['goodjets'] = add_jec_variables(events.goodjets, events.fixedGridRhoFastjetAll)
        events['Jet'] = add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll)

        # # Boson storage
        if dset_type == 'mc':
            if type(pdgId) == int:
                genparts = events.GenPart[
                        (abs(events.GenPart.pdgId) == pdgId)
                        & events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy'])
                    ]
            elif type(pdgId) == str:
                genparts = events.GenPart[
                        ((abs(events.GenPart.pdgId) == 24)|(events.GenPart.pdgId == 23))
                        & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
                    ]
            else:
                genparts = events.GenPart
            matchedBoson = events.FatJet.nearest(genparts, threshold=0.2)
            events['matchedBoson'] =  matchedBoson[~ak.is_none(matchedBoson, axis=1)]
            events['matchedBoson', 'children'] = events.matchedBoson.children

        # Write parquet file with Dask Awkward
        path = f"/project01/ndcms/cmoore24/skims/analysis_skims/{year}/{lep_region}/{dset_type}/{dataset}" #Edit this for parquet file destination!
        if (match==False) and (dset_type == 'mc'):
            path = f"/project01/ndcms/cmoore24/skims/analysis_skims/{year}/{lep_region}/{dset_type}_unmatched/{dataset}"
            
        skim_task = dak.to_parquet(
            events,
            path, ##Change this to where you'd like the output to be written
            compute=False,
            write_metadata=False, 
            extensionarray=False,
        )
        return skim_task


    ##### Begin compputing the delayed skim task ##### 

    # Compute only a subset of the datasets available in samples_ready.json
    subset = {}
    batch = {}
    # to_skim = 'qcd' # Use this string to choose the subsample. Name must be found in samples_ready.json
    for to_skim in samples_ready:
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
        
            for i in range(len(files)):
                batch[to_skim]['files'][keys[i]] = files[keys[i]]
        else:
            continue

    # Create the dictionary of tasks to compute
    tasks = dataset_tools.apply_to_fileset(
        analysis,
        # samples_ready, ## Run over all subsamples in input_datasets.json
        batch, ## Run over only the subsample specified as the "to_skim" string
        uproot_options={"allow_read_errors_with_report": False},
        schemaclass = PFNanoAODSchema,
    )

    # Call dask.compute with TaskVine
    print('start compute')
    computed = dask.compute(
            tasks,
            scheduler=m.get,
            # scheduling_mode="breadth-first",
            worker_transfers=True,
            resources={"cores": 1},
            task_mode="function-calls",
            lib_resources={'cores': 1, 'slots': 1},
            prune_depth=2, 
            optimize_graph=True,
        )

    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))