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
from ndcctools.taskvine import DaskVine
import time

full_start = time.time()

if __name__ == "__main__":
    m = DaskVine(
        [9123, 9128],
        name=f"{os.environ['USER']}-hgg",
        run_info_path=f"/project01/ndcms/{os.environ['USER']}/vine-run-info",
    )

    available_functions = [
        "Color_Ring",
        "Color_Ring_Var",
        "D2",
    ]

    enabled_functions = set()
    enabled_functions.update(["Color_Ring"])
    print(enabled_functions)

    warnings.filterwarnings("ignore", "Found duplicate branch")
    warnings.filterwarnings("ignore", "Missing cross-reference index for")
    warnings.filterwarnings("ignore", "dcut")
    warnings.filterwarnings("ignore", "Please ensure")
    warnings.filterwarnings("ignore", "The necessary")
    warnings.filterwarnings("ignore",  module="coffea.*")


    try:
        with open("data_dv4.pkl", "rb") as fr:
            datasets = pickle.load(fr)
    except Exception as e:
        print(f"Could not read data_dv4.pkl: {e}, reconstructing...")
        datasets = {
            "hgg": {"path": "signal/hgg", "label": "Hgg"},
            "hbb": {"path": "signal/hbb", "label": "Hbb"},
            "q173": {"path": "qcd/170to300", "label": "QCD_Pt_170to300"},
            "q347": {"path": "qcd/300to470", "label": "QCD_Pt_300to470"},
            "q476": {"path": "qcd/470to600", "label": "QCD_Pt_470to600"},
            "q68": {"path": "qcd/600to800", "label": "QCD_Pt_600to800"},
            "q810": {"path": "qcd/800to1000", "label": "QCD_Pt_800to1000"},
            "q1014": {"path": "qcd/1000to1400", "label": "QCD_Pt_1000to1400"},
            "q1418": {"path": "qcd/1400to1800", "label": "QCD_Pt_1400to1800"},
            "q1824": {"path": "qcd/1800to2400", "label": "QCD_Pt_1800to2400"},
            "q2432": {"path": "qcd/2400to3200", "label": "QCD_Pt_2400to3200"},
            "q32Inf": {"path": "qcd/3200toInf", "label": "QCD_Pt_3200toInf"},
        }

        path_root = "/project01/ndcms/cmoore24"
        for name, info in datasets.items():
            info["files"] = os.listdir(f"{path_root}/{info['path']}")

        with open("data_dv4.pkl", "wb") as fw:
            pickle.dump(datasets, fw)

    source = "/project01/ndcms/cmoore24"
    events = {}
    for name, info in datasets.items():
        events[name] = NanoEventsFactory.from_root(
            {f"{source}/{info['path']}/{fn}": "/Events" for fn in info["files"]},
            schemaclass=PFNanoAODSchema,
            # uproot_options={"chunks_per_file":1},
            metadata={"dataset": info["label"]},
        ).events()

    def num_subjets(fatjet, cluster):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.inclusive_jets()
        num = ak.num(subjets, axis=1)
        return num

    def color_ring(fatjet, cluster, variant=False):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.inclusive_jets()
        vec = ak.zip(
            {
                "x": subjets.px,
                "y": subjets.py,
                "z": subjets.pz,
                "t": subjets.E,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        vec = ak.pad_none(vec, 3)
        vec["norm3"] = np.sqrt(vec.dot(vec))
        vec["idx"] = ak.local_index(vec)
        i, j, k = ak.unzip(ak.combinations(vec, 3))
        #best = ak.argmin(abs((i + j + k).mass - 125), axis=1, keepdims=True)
        best = ak.argmax(abs((i + j + k).mass), axis=1, keepdims=True)
        order_check = ak.concatenate([i[best].pt, j[best].pt, k[best].pt], axis=1)
        largest = ak.argmax(order_check, axis=1, keepdims=True)
        smallest = ak.argmin(order_check, axis=1, keepdims=True)
        leading_particles = ak.concatenate([i[best], j[best], k[best]], axis=1)
        leg1 = leading_particles[largest]
        leg3 = leading_particles[smallest]
        leg2 = leading_particles[
            (leading_particles.idx != ak.flatten(leg1.idx))
            & (leading_particles.idx != ak.flatten(leg3.idx))
        ]
        leg1 = ak.firsts(leg1)
        leg2 = ak.firsts(leg2)
        leg3 = ak.firsts(leg3)
        a12 = np.arccos(leg1.dot(leg2) / (leg1.norm3 * leg2.norm3))
        a13 = np.arccos(leg1.dot(leg3) / (leg1.norm3 * leg3.norm3))
        a23 = np.arccos(leg2.dot(leg3) / (leg2.norm3 * leg3.norm3))
        if not variant:
            color_ring = (a13**2 + a23**2) / (a12**2)
        else:
            color_ring = a13**2 + a23**2 - a12**2
        return color_ring

    def d2_calc(fatjet):
        jetdef = fastjet.JetDefinition(
            fastjet.cambridge_algorithm, 0.8
        )
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop = cluster.exclusive_jets_softdrop_grooming()
        softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
        d2 = softdrop_cluster.exclusive_jets_energy_correlator(func="D2")
        return d2

    class MyProcessor(processor.ProcessorABC):
        def __init__(self):
            pass

        def process(self, events):
            dataset = events.metadata["dataset"]
            computations = {"entries": ak.num(events.event, axis=0)}

            fatjet = events.FatJet

            if "QCD" in dataset:
                print("background")
                cut = (
                    (fatjet.pt > 200)
                    & (fatjet.pt < 2500)
                    & (fatjet.mass > 50)
                    & (fatjet.mass < 200)
                    & (abs(fatjet.eta) < 2.5)
                    # & (fatjet.btagDDBvLV2 > 0.20)
                )

            else:
                print("signal")
                genhiggs = events.GenPart[
                    (events.GenPart.pdgId == 25)
                    & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])
                ]
                parents = events.FatJet.nearest(genhiggs, threshold=0.2)
                higgs_jets = ~ak.is_none(parents, axis=1)

                cut = (
                    (fatjet.pt > 200)
                    & (fatjet.pt < 2500)
                    & (fatjet.mass > 50)
                    & (fatjet.mass < 200)
                    & (abs(fatjet.eta) < 2.5)
                    & (higgs_jets)  
                    # & (fatjet.btagDDBvLV2 > 0.20)
                )

            boosted_fatjet = fatjet[cut]
            boosted_fatjet.constituents.pf["pt"] = (
                boosted_fatjet.constituents.pf.pt
                * boosted_fatjet.constituents.pf.puppiWeight
            )

            num_sub = ak.unflatten(num_subjets(boosted_fatjet, cluster=0.2), counts=ak.num(boosted_fatjet))
            boosted_fatjet['num_subjets'] = num_sub
            sub_cut = (boosted_fatjet.num_subjets >= 3)
            boosted_fatjet = boosted_fatjet[sub_cut]

            # if "Color_Ring" in enabled_functions:
            #     uf_cr = ak.unflatten(
            #         color_ring(boosted_fatjet, cluster=0.3), counts=ak.num(boosted_fatjet)
            #     )
            #     boosted_fatjet["color_ring"] = uf_cr
            #     hcr = dhist.Hist.new.Reg(
            #         100, 0.25, 10, name="color_ring", label="Color_Ring",
            #         overflow=False, underflow=False,
            #     ).Weight()
            #     fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
            #     hcr.fill(color_ring=fill_cr)
            #     computations["Color_Ring_test"] = hcr

            if "Color_Ring" in enabled_functions:
                uf_cr = ak.unflatten(
                     color_ring(boosted_fatjet, cluster=0.2), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["color_ring"] = uf_cr
                fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 0)
                multi_axis = (
                    dhist.Hist.new
                    .Reg(40, 0, 3, name="Color_Ring", label="Color_Ring", overflow=False, underflow=False)
                    .Reg(40, 150, 2500, name="PT", label="PT", overflow=False, underflow=False)
                    .Reg(40, 50, 150, name="Mass", label="Mass", overflow=False, underflow=False)
                    .Reg(40, 50, 150, name="SDMass", label="SDMass", overflow=False, underflow=False)
                    .Weight()
                    .fill(Color_Ring=fill_cr, 
                          PT=ak.flatten(boosted_fatjet.pt),
                          Mass=ak.flatten(boosted_fatjet.mass),
                          SDMass=ak.flatten(boosted_fatjet.msoftdrop)

                         )
                )
                computations["Color_Ring"] = multi_axis

            if "Color_Ring_Var" in enabled_functions:
                uf_cr_var = ak.unflatten(
                    color_ring(boosted_fatjet, variant=True),
                    counts=ak.num(boosted_fatjet),
                )
                boosted_fatjet["color_ring_var"] = uf_cr_var
                hcr_var = dhist.Hist.new.Reg(
                100, 0, 3, name="color_ring_var", label="Color_Ring_Var"
                ).Weight()
                fill_cr_var = ak.fill_none(ak.flatten(boosted_fatjet.color_ring_var), 0)
                hcr_var.fill(color_ring_var=fill_cr_var)
                computations["Color_Ring_Var"] = hcr_var

            if "D2" in enabled_functions:
                d2 = ak.unflatten(
                    d2_calc(boosted_fatjet), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["d2b1"] = d2
                d2b1 = dhist.Hist.new.Reg(40, 0, 3, name="D2B1", label="D2B1", overflow=False, underflow=False
                ).Weight()
                d2b1.fill(D2B1=ak.flatten(boosted_fatjet.d2b1))
                computations["D2"] = d2b1

            return {dataset: computations}
                
        def postprocess(self, accumulator):
            pass
        
    start = time.time()
    result = {}
    result["Hgg"] = MyProcessor().process(events["hgg"])
    print("hbb")
    result["Hbb"] = MyProcessor().process(events["hbb"])
    print("170")
    result["QCD_Pt_170to300_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q173"]
    )
    print("300")
    result["QCD_Pt_300to470_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q347"]
    )
    print("470")
    result["QCD_Pt_470to600_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q476"]
    )
    print("600")
    result["QCD_Pt_600to800_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q68"]
    )
    print("800")
    result["QCD_Pt_800to1000_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q810"]
    )
    print("1000")
    result["QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q1014"]
    )
    print("1400")
    result["QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q1418"]
    )
    print("1800")
    result["QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q1824"]
    )
    print("2400")
    result["QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q2432"]
    )
    print("3200")
    result["QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8"] = MyProcessor().process(
        events["q32Inf"]
    )
    stop = time.time()
    print(stop - start)

    print("computing")
    computed = dask.compute(
        result,
        scheduler=m.get,
        resources={"cores": 1},
        resources_mode=None,
        lazy_transfers=True,
        #task_mode="function_calls",
        lib_resources={'cores': 12, 'slots': 12},
    )

    name = '../outputs/cr_investigations/multi_var_hists/cambridge02_pt_mass_argmax.pkl'
    with open(name, 'wb') as f:
        pickle.dump(computed, f)

    
    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))
