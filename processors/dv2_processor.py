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

    with open('filelists/hgg_files.txt', 'r') as f:
        hgg_files = [line.strip() for line in f]
    with open('filelists/hbb_files.txt', 'r') as f:
        hbb_files = [line.strip() for line in f]
    with open('filelists/q347_files.txt', 'r') as f:
        q347_files = [line.strip() for line in f]
    with open('filelists/q476_files.txt', 'r') as f:
        q476_files = [line.strip() for line in f]
    with open('filelists/q68_files.txt', 'r') as f:
        q68_files = [line.strip() for line in f]
    with open('filelists/q810_files.txt', 'r') as f:
        q810_files = [line.strip() for line in f]
    with open('filelists/q1014_files.txt', 'r') as f:
        q1014_files = [line.strip() for line in f]
    with open('filelists/q1418_files.txt', 'r') as f:
        q1418_files = [line.strip() for line in f]
    with open('filelists/q1824_files.txt', 'r') as f:
        q1824_files = [line.strip() for line in f]
    with open('filelists/q2432_files.txt', 'r') as f:
        q2432_files = [line.strip() for line in f]
    with open('filelists/q32inf_files.txt', 'r') as f:
        q32inf_files = [line.strip() for line in f]
    

    hgg = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/signal/hgg/' + fn: "/Events"} for fn in hgg_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "Hgg"},
    ).events()
    
    hbb = NanoEventsFactory.from_root(
        [{'/project01/ndcms/cmoore24/signal/hbb/' + fn: "/Events"} for fn in hbb_files],
        permit_dask=True,
        schemaclass=PFNanoAODSchema,
        metadata={"dataset": "Hbb"},
    ).events()

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

    def color_ring(fatjet, variant=False, groomed=False):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        if groomed==True:
            jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
            cluster = fastjet.ClusterSequence(pf, jetdef)
            softdrop = cluster.exclusive_jets_softdrop_grooming()
            pf = softdrop.constituents
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.2)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        #subjets = cluster.exclusive_subjets_up_to(data=cluster.exclusive_jets(n_jets=1), nsub=3) #legacy
        subjets = cluster.inclusive_jets()
        vec = ak.zip({
            "x": subjets.px,
            "y": subjets.py,
            "z": subjets.pz,
            "t": subjets.E,
            },
            with_name = "LorentzVector",
            behavior=vector.behavior,
            )
        vec = ak.pad_none(vec, 3)
        vec["norm3"] = np.sqrt(vec.dot(vec))
        vec["idx"] = ak.local_index(vec)
        # i, j = ak.unzip(ak.combinations(vec, 2))
        # best = ak.argmax((i + j).mass, axis=1, keepdims=True)
        # leg1, leg2 = ak.firsts(i[best]), ak.firsts(j[best])
        # leg3 = ak.firsts(vec[(vec.idx != leg1.idx) & (vec.idx != leg2.idx)]) #new
        i, j, k = ak.unzip(ak.combinations(vec, 3))
        best = ak.argmin(abs((i + j + k).mass - 125), axis=1, keepdims=True)
        order_check = ak.concatenate([i[best].mass, j[best].mass, k[best].mass], axis=1)
        largest = ak.argmax(order_check, axis=1, keepdims=True)
        smallest = ak.argmin(order_check, axis=1, keepdims=True)
        leading_particles = ak.concatenate([i[best], j[best], k[best]], axis=1)
        leg1 = leading_particles[largest]
        leg3 = leading_particles[smallest]
        leg2 = leading_particles[(leading_particles.idx != ak.flatten(leg1.idx)) & (leading_particles.idx != ak.flatten(leg3.idx))]
        leg1 = ak.firsts(leg1)
        leg2 = ak.firsts(leg2)
        leg3 = ak.firsts(leg3)
        a12 = np.arccos(leg1.dot(leg2) / (leg1.norm3 * leg2.norm3))
        a13 = np.arccos(leg1.dot(leg3) / (leg1.norm3 * leg3.norm3))
        a23 = np.arccos(leg2.dot(leg3) / (leg2.norm3 * leg3.norm3))
        if variant == False:
            color_ring = ((a13**2 + a23**2)/(a12**2))
        else: 
            color_ring = a13**2 + a23**2 - a12**2
        return color_ring

    def d2_calc(fatjet):
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8) # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        d2 = softdrop_cluster.exclusive_jets_energy_correlator(func='D2')
        return d2

    def n4_calc(fatjet):
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8) # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        e25 = softdrop_cluster.exclusive_jets_energy_correlator(func='generalized', angles=2, npoint=5)
        e14 = softdrop_cluster.exclusive_jets_energy_correlator(func='generalized', angles=1, npoint=4)
        n4 = e25/(e14**2)
        return n4

    def d3_calc(fatjet):
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8) # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        e4 = softdrop_cluster.exclusive_jets_energy_correlator(func='generic', normalized=True, npoint=4)
        e2 = softdrop_cluster.exclusive_jets_energy_correlator(func='generic', normalized=True, npoint=2)
        e3 = softdrop_cluster.exclusive_jets_energy_correlator(func='generic', normalized=True, npoint=3)
        d3 = (e4*(e2**3))/(e3**3)
        return d3

    def u_calc(fatjet, n):
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8) # make this C/A at 0.8
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        if n==1:
            u = softdrop_cluster.exclusive_jets_energy_correlator(func='u1')
        if n==2:
            u = softdrop_cluster.exclusive_jets_energy_correlator(func='u2')
        if n==3:
            u = softdrop_cluster.exclusive_jets_energy_correlator(func='u3')
        return u  

    class MyProcessor(processor.ProcessorABC):
        
        def __init__(self):
            pass
    
        
        def process(self, events):
            dataset = events.metadata['dataset']
            
            fatjet = events.FatJet

            if 'QCD' in dataset:
                print('background')
                cut = ((fatjet.pt > 300) & (fatjet.msoftdrop > 110) & 
                   (fatjet.msoftdrop < 140) & (abs(fatjet.eta) < 2.5)) #& (fatjet.btagDDBvLV2 > 0.20)

            else:
                print('signal')
                genhiggs = (events.GenPart[
                    (events.GenPart.pdgId==25)
                    & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
                ])
                parents = events.FatJet.nearest(genhiggs, threshold=0.1)
                higgs_jets = ~ak.is_none(parents, axis=1)
                
                
                cut = ((fatjet.pt > 300) & (fatjet.msoftdrop > 110) & 
                       (fatjet.msoftdrop < 140) & (abs(fatjet.eta) < 2.5)) & (higgs_jets) #& (fatjet.btagDDBvLV2 > 0.20)
            
            boosted_fatjet = fatjet[cut]
            boosted_fatjet.constituents.pf['pt'] = boosted_fatjet.constituents.pf.pt*boosted_fatjet.constituents.pf.puppiWeight
            
            # uf_cr = ak.unflatten(color_ring(boosted_fatjet), counts=ak.num(boosted_fatjet))
            # uf_cr_groomed = ak.unflatten(color_ring(boosted_fatjet, groomed=True), counts=ak.num(boosted_fatjet))
            # uf_cr_var = ak.unflatten(color_ring(boosted_fatjet, variant=True), counts=ak.num(boosted_fatjet))
            # d2 = ak.unflatten(d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            n4 = ak.unflatten(n4_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            # d3 = ak.unflatten(d3_calc(boosted_fatjet), counts=ak.num(boosted_fatjet))
            # u1 = ak.unflatten(u_calc(boosted_fatjet, n=1), counts=ak.num(boosted_fatjet))
            # u2 = ak.unflatten(u_calc(boosted_fatjet, n=2), counts=ak.num(boosted_fatjet))
            # u3 = ak.unflatten(u_calc(boosted_fatjet, n=3), counts=ak.num(boosted_fatjet))
            # boosted_fatjet['color_ring'] = uf_cr
            # boosted_fatjet['color_ring_groomed'] = uf_cr_groomed
            #boosted_fatjet['color_ring_var'] = uf_cr_var
            # boosted_fatjet['d2b1'] = d2
            boosted_fatjet['n4b1'] = n4
            # boosted_fatjet['d3b1'] = d3
            # boosted_fatjet['u1b1'] = u1
            # boosted_fatjet['u2b1'] = u2
            # boosted_fatjet['u3b1'] = u3
            
            # hcr = (
            #     dhist.Hist.new
            #     .Reg(40, 0.5, 3.5, name='color_ring', label='Color_Ring')
            #     .Weight()
            # )

            # hcr_groomed = (
            #     dhist.Hist.new
            #     .Reg(40, 0.5, 3.5, name='color_ring_groomed', label='Color_Ring_Groomed')
            #     .Weight()
            # )

            # hcr_var = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 3, name='color_ring_var', label='Color_Ring_Var')
            #     .Weight()
            # )
    
            # d2b1 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 3, name='D2B1', label='D2B1')
            #     .Weight()
            # )

            # d3b1 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 3, name='D3B1', label='D3B1')
            #     .Weight()
            # )

            n4b1 = (
                dhist.Hist.new
                .Reg(40, 0, 35, name='N4B1', label='N4B1')
                .Weight()
            )

            # u1b1 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, .3, name='U1B1', label='U1B1')
            #     .Weight()
            # )

            # u2b1 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 0.05, name='U2B1', label='U2B1')
            #     .Weight()
            # )

            # u3b1 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 0.05, name='U3B1', label='U3B1')
            #     .Weight()
            # )
            
            # cmssw_n2 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 0.5, name='cmssw_n2', label='CMSSW_N2')
            #     .Weight()
            # )
            
            # cmssw_n3 = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 3, name='cmssw_n3', label='CMSSW_N3')
            #     .Weight()
            # )
            
            # ncons = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 200, name='constituents', label='nConstituents')
            #     .Weight()
            # )
            
            # mass = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 250, name='mass', label='Mass')
            #     .Weight()
            # )
            
            # sdmass = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 250, name='sdmass', label='SDmass')
            #     .Weight()
            # )
    
            # btag = (
            #     dhist.Hist.new
            #     .Reg(40, 0, 1, name='Btag', label='Btag')
            #     .Weight()
            # )
            
            # fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
            # hcr.fill(color_ring=fill_cr)
            # fill_cr_groomed = ak.fill_none(ak.flatten(boosted_fatjet.color_ring_groomed), 0)
            # hcr_groomed.fill(color_ring_groomed=fill_cr_groomed)
            #fill_cr_var = ak.fill_none(ak.flatten(boosted_fatjet.color_ring_var), 0)
            #hcr_var.fill(color_ring_var=fill_cr_var)
            # d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
            # d3b1.fill(D3B1=ak.flatten(boosted_fatjet.d3b1))
            n4b1.fill(N4B1=ak.flatten(boosted_fatjet.n4b1))
            # u1b1.fill(U1B1=ak.flatten(boosted_fatjet.u1b1))
            # u2b1.fill(U2B1=ak.flatten(boosted_fatjet.u2b1))
            # u3b1.fill(U3B1=ak.flatten(boosted_fatjet.u3b1))
            # cmssw_n2.fill(cmssw_n2=ak.flatten(boosted_fatjet.n2b1))
            # cmssw_n3.fill(cmssw_n3=ak.flatten(boosted_fatjet.n3b1))
            # ncons.fill(constituents=ak.flatten(boosted_fatjet.nConstituents))
            # mass.fill(mass=ak.flatten(boosted_fatjet.mass))
            # sdmass.fill(sdmass=ak.flatten(boosted_fatjet.msoftdrop))
            # btag.fill(Btag=ak.flatten(boosted_fatjet.btagDDBvLV2))

            
            return {
                dataset: {
                    "entries": ak.count(events.event, axis=None),
                    # "Color_Ring": hcr,
                    # "Color_Ring_Groomed": hcr_groomed,
                    #"Color_Ring_Var": hcr_var,
                    # "N2": cmssw_n2,
                    # "N3": cmssw_n3,
                    # "nConstituents": ncons,
                    # "Mass": mass,
                    # "SDmass": sdmass,
                    # "Btag": btag,
                    # "D2": d2b1,
                    # "D3": d3b1,
                    "N4": n4b1,
                    # "U1": u1b1,
                    # "U2": u2b1,
                    # "U3": u3b1,
                }
            }
        
        def postprocess(self, accumulator):
            pass

    start = time.time()
    result = {}
    result['Hgg'] = MyProcessor().process(hgg)
    print('hbb')
    result['Hbb'] = MyProcessor().process(hbb)
    print('300')
    result['QCD_Pt_300to470_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q347)
    print('470')
    result['QCD_Pt_470to600_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q476)
    print('600')
    result['QCD_Pt_600to800_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q68)
    print('800')
    result['QCD_Pt_800to1000_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q810)
    print('1000')
    result['QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q1014)
    print('1400')
    result['QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q1418)
    print('1800')
    result['QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q1824)
    print('2400')
    result['QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q2432)
    print('3200')
    result['QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8'] = MyProcessor().process(q32Inf)
    stop = time.time()
    print(stop-start)

    print('computing')
    computed = dask.compute(result, scheduler=m.get, resources={"cores": 1}, resources_mode=None, lazy_transfers=True)
    with open('outputs/n4_not_btagged.pkl', 'wb') as f:
        pickle.dump(computed, f)


full_stop = time.time()
print('full run time is ' + str((full_stop - full_start)/60))
