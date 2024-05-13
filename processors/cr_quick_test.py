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
            #"q173": {"path": "qcd/170to300", "label": "QCD_Pt_170to300"},
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

    def num_subjets(fatjet, cluster_val):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster_val)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        subjets = cluster.inclusive_jets()
        num = ak.num(subjets, axis=1)
        return num

    #skim version
    # def color_ring(fatjet, cluster_val):
    #     pf = ak.flatten(fatjet.constituents.pf, axis=1)
    #     jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster_val)
    #     cluster = fastjet.ClusterSequence(pf, jetdef)
    #     #subjets = cluster.exclusive_jets(n_jets=3)
    #     subjets = cluster.inclusive_jets()
    #     vec = ak.zip(
    #         {
    #             "x": subjets.px,
    #             "y": subjets.py,
    #             "z": subjets.pz,
    #             "t": subjets.E,
    #         },
    #         with_name="LorentzVector",
    #         behavior=vector.behavior,
    #     )
    #     vec = ak.pad_none(vec, 3)
    #     vec["norm3"] = np.sqrt(vec.dot(vec))
    #     vec["idx"] = ak.local_index(vec)
    #     i, j = ak.unzip(ak.combinations(vec, 2))
    #     best = ak.argmax(abs((i + j).pt), axis=1, keepdims=True)
    #     order_check = ak.concatenate([i[best].pt, j[best].pt], axis=1)
    #     leading = ak.argmax(order_check, axis=1, keepdims=True)
    #     subleading = ak.argmin(order_check, axis=1, keepdims=True)
    #     leading_particles = ak.concatenate([i[best], j[best]], axis=1)
    #     cut = ((vec.idx != ak.firsts(leading_particles.idx)) & 
    #        (vec.idx != ak.firsts(ak.sort(leading_particles.idx, ascending=False)))
    #       )
    #     everything_else = vec[cut]
    #     total_pt = ak.sum(vec.pt, axis=1)
    #     everything_else['momentum_fraction'] = (everything_else.pt)/total_pt
    #     everything_else['weighted_eta'] = everything_else.eta * everything_else.momentum_fraction
    #     everything_else['weighted_phi'] = everything_else.phi * everything_else.momentum_fraction
    #     weighted_average_eta = ak.sum(everything_else.weighted_eta, axis=1)/ak.num(everything_else, axis=1)
    #     weighted_average_phi = ak.sum(everything_else.weighted_phi, axis=1)/ak.num(everything_else, axis=1)
    #     leg1 = leading_particles[leading]
    #     leg2 = leading_particles[subleading]
    #     leg1 = ak.firsts(leg1)
    #     leg2 = ak.firsts(leg2)
    #     a13 = ((((leg1.eta * (leg1.pt/total_pt)
    #              ) - weighted_average_eta)**2) + (((leg1.phi * (leg1.pt/total_pt)
    #                                       ) - weighted_average_phi)**2))
    #     a23 = ((((leg2.eta * (leg2.pt/total_pt)
    #              ) - weighted_average_eta)**2) + (((leg2.phi * (leg2.pt/total_pt)
    #                                       ) - weighted_average_phi)**2))
    #     a12 = ((((leg1.eta * (leg1.pt/total_pt)
    #              ) - (leg2.eta * (leg2.pt/total_pt)
    #                  ))**2) + (((leg1.phi * (leg1.pt/total_pt)
    #                             ) - (leg2.phi * (leg2.pt/total_pt)
    #                                 ))**2))
    #     color_ring = (a13 + a23) / (a12)
    
    #     return color_ring

    #old version
    def color_ring(fatjet, cluster_val):
        pf = ak.flatten(fatjet.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, cluster_val)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        #subjets = cluster.exclusive_jets(n_jets=3)
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
        i, j = ak.unzip(ak.combinations(vec, 2))
        best = ak.argmax(abs((i + j).pt), axis=1, keepdims=True)
        order_check = ak.concatenate([i[best].pt, j[best].pt], axis=1)
        leading = ak.argmax(order_check, axis=1, keepdims=True)
        subleading = ak.argmin(order_check, axis=1, keepdims=True)
        leading_particles = ak.concatenate([i[best], j[best]], axis=1)
        cut = ((vec.idx != ak.firsts(leading_particles.idx)) & 
           (vec.idx != ak.firsts(ak.sort(leading_particles.idx, ascending=False)))
          )
        everything_else = vec[cut]
        total_pt = ak.sum(vec.pt, axis=1)
        everything_else['momentum_fraction'] = (everything_else.pt)/total_pt
        everything_else['weighted_eta'] = everything_else.eta * everything_else.momentum_fraction
        everything_else['weighted_phi'] = everything_else.phi * everything_else.momentum_fraction
        weighted_average_eta = ak.sum(everything_else.weighted_eta, axis=1)/ak.num(everything_else, axis=1)
        average_eta = ak.sum(everything_else.eta, axis=1)/ak.num(everything_else, axis=1)
        weighted_average_phi = ak.sum(everything_else.weighted_phi, axis=1)/ak.num(everything_else, axis=1)
        average_phi = ak.sum(everything_else.phi, axis=1)/ak.num(everything_else, axis=1)
        leg1 = leading_particles[leading]
        leg2 = leading_particles[subleading]
        leg1 = ak.firsts(leg1)
        leg2 = ak.firsts(leg2)
        a13 = ((((leg1.eta * (leg1.pt/total_pt)
                 ) - average_eta)**2) + (((leg1.phi * (leg1.pt/total_pt)
                                          ) - weighted_average_phi)**2))
        a23 = ((((leg2.eta * (leg2.pt/total_pt)
                 ) - average_eta)**2) + (((leg2.phi * (leg2.pt/total_pt)
                                          ) - weighted_average_phi)**2))
        a12 = ((((leg1.eta * (leg1.pt/total_pt)
                 ) - (leg2.eta * (leg2.pt/total_pt)
                     ))**2) + (((leg1.phi * (leg1.pt/total_pt)
                                ) - (leg2.phi * (leg2.pt/total_pt)
                                    ))**2))
        color_ring = (a13 + a23) / (a12)
    
        return color_ring

    class MyProcessor(processor.ProcessorABC):
        def __init__(self):
            pass

        def process(self, events):
            dataset = events.metadata["dataset"]
            computations = {"entries": ak.num(events.event, axis=0)}
            
            cut_to_fix_softdrop = (ak.num(events.FatJet.constituents.pf, axis=2) > 0)

            events['PFCands', 'pt'] = (
                events.PFCands.pt
                * events.PFCands.puppiWeight
            )

            fatjet = events.FatJet

            if "QCD" in dataset:
                print("background")
                cut = (
                    (fatjet.pt > 250)
                    & (fatjet.mass > 50)
                    & (fatjet.mass < 200)
                    & (abs(fatjet.eta) < 2.5)
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

            num_sub = ak.unflatten(num_subjets(boosted_fatjet, cluster_val=0.4), counts=ak.num(boosted_fatjet))
            boosted_fatjet['num_subjets'] = num_sub
            sub_cut = (boosted_fatjet.num_subjets >= 3)
            boosted_fatjet = boosted_fatjet[sub_cut]

            if "Color_Ring" in enabled_functions:
                uf_cr = ak.unflatten(
                     color_ring(boosted_fatjet, cluster_val=0.4), counts=ak.num(boosted_fatjet)
                )
                boosted_fatjet["color_ring"] = uf_cr
                fill_cr = ak.fill_none(ak.flatten(boosted_fatjet.color_ring), 10)
                multi_axis = (
                    dhist.Hist.new
                    .Reg(40, 0, 4, name="Color_Ring", label="Color_Ring", overflow=False, underflow=False)
                    .Reg(40, 150, 2500, name="PT", label="PT", overflow=False, underflow=False)
                    .Reg(40, 50, 150, name="Mass", label="Mass", overflow=False, underflow=False)
                    .Weight()
                    .fill(
                          Color_Ring=fill_cr, 
                          PT=ak.flatten(boosted_fatjet.pt),
                          Mass=ak.flatten(boosted_fatjet.mass),
                         )
                )
                computations["Color_Ring"] = multi_axis

            return {dataset: computations}
                
        def postprocess(self, accumulator):
            pass

    start = time.time()
    result = {}
    result["Hgg"] = MyProcessor().process(events["hgg"])
    print("hbb")
    result["Hbb"] = MyProcessor().process(events["hbb"])
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

    name = '../outputs/cr_investigations/multi_var_hists/old_cr_quick_check.pkl'
    with open(name, 'wb') as f:
        pickle.dump(computed, f)

    
    full_stop = time.time()
    print('full run time is ' + str((full_stop - full_start)/60))