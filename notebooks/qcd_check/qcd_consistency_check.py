from coffea.nanoevents import NanoEventsFactory, BaseSchema, PFNanoAODSchema
import json
import fastjet
import numpy as np
import awkward as ak
from coffea import processor
import hist
import coffea.nanoevents.methods.vector as vector
import warnings
import hist.dask as dhist
import dask
import pickle
import os
import distributed
from ndcctools.taskvine import DaskVine
import time

full_start = time.time()

if __name__ == '__main__':
    m = DaskVine([9123,9128], name="hgg", run_info_path='/project01/ndcms/cmoore24/vine-run-info')

    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "The necessary")

    q347_files = os.listdir('/project01/ndcms/cmoore24/qcd/300to470')
    q476_files = os.listdir('/project01/ndcms/cmoore24/qcd/470to600')
    q68_files = os.listdir('/project01/ndcms/cmoore24/qcd/600to800')
    q810_files = os.listdir('/project01/ndcms/cmoore24/qcd/800to1000')
    q1014_files = os.listdir('/project01/ndcms/cmoore24/qcd/1000to1400')
    q1418_files = os.listdir('/project01/ndcms/cmoore24/qcd/1400to1800')
    q1824_files = os.listdir('/project01/ndcms/cmoore24/qcd/1800to2400')
    q2432_files = os.listdir('/project01/ndcms/cmoore24/qcd/2400to3200')
    q32inf_files = os.listdir('/project01/ndcms/cmoore24/qcd/3200toInf')

    q347 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/300to470/' + fn: "/Events"} for fn in q347_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_300to470"},
    ).events()
    
    q476 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/470to600/' + fn: "/Events"} for fn in q476_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_470to600"},
    ).events()
    
    q68 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/600to800/' + fn: "/Events"} for fn in q68_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_600to800"},
    ).events()
    
    q810 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/800to1000/' + fn: "/Events"} for fn in q810_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_800to1000"},
    ).events()
    
    q1014 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/1000to1400/' + fn: "/Events"} for fn in q1014_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_1000to1400"},
    ).events()
    
    q1418 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/1400to1800/' + fn: "/Events"} for fn in q1418_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_1400to1800"},
    ).events()
    
    q1824 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/1800to2400/' + fn: "/Events"} for fn in q1824_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_1800to2400"},
    ).events()
    
    q2432 = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/2400to3200/' + fn: "/Events"} for fn in q2432_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_2400to3200"},
    ).events()
    
    q32Inf = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/qcd/3200toInf/' + fn: "/Events"} for fn in q32inf_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "QCD_Pt_3200toInf"},
    ).events()

    class MyProcessor_Background(processor.ProcessorABC):
    
        def __init__(self):
            pass
        
        def process(self, events):
            dataset = events.metadata['dataset']
            
            fatjet = events.FatJet
            cut = ((fatjet.pt > 0) #& (fatjet.msoftdrop > 110) & 
                   #(fatjet.msoftdrop < 140) & (abs(fatjet.eta) < 2.5) #& (higgs_jets)
                  )
            boosted_fatjet = fatjet[cut]
            
            jet_pt = (
                dhist.Hist.new
                .Reg(75, 200, 5000, name='jet_pt', label='Jet_Pt')
                .Weight()
            )
            
            jet_pt.fill(jet_pt=ak.flatten(boosted_fatjet.pt))
            
            return {
                dataset: {
                    "entries": ak.count(events.event, axis=None),
                    "Jet_Pt":jet_pt,
                }
            }
        
        def postprocess(self, accumulator):
            pass

    start = time.time()
    result = {}
    print('300')
    result['QCD_Pt_300to470_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q347)
    print('470')
    result['QCD_Pt_470to600_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q476)
    print('600')
    result['QCD_Pt_600to800_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q68)
    print('800')
    result['QCD_Pt_800to1000_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q810)
    print('1000')
    result['QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q1014)
    print('1400')
    result['QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q1418)
    print('1800')
    result['QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q1824)
    print('2400')
    result['QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q2432)
    print('3200')
    result['QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8'] = MyProcessor_Background().process(q32Inf)
    stop = time.time()
    print(stop-start)

    print('computing')
    computed = dask.compute(result, scheduler=m.get, resources={"cores": 1}, resources_mode=None, lazy_transfers=True)
    with open('./qcd_pt.pkl', 'wb') as f:
        pickle.dump(computed, f)


full_stop = time.time()
print('full run time is ' + str((full_stop - full_start)/60))