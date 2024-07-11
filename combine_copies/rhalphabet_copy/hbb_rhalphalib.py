from __future__ import print_function, division
import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import ROOT

rl.util.install_roofit_helpers()


def load_hist(filename):
    with open(f'hists/hbb_hists/{filename}', 'rb') as f:
        hist = pickle.load(f)
    return hist


def expo_sample(norm, scale, obs):
    cdf = scipy.stats.expon.cdf(scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def gaus_sample(norm, loc, scale, obs):
    cdf = scipy.stats.norm.cdf(loc=loc, scale=scale, x=obs.binning) * norm
    return (np.diff(cdf), obs.binning, obs.name)


def test_rhalphabet(tmpdir):
    throwPoisson = False

    jec = rl.NuisanceParameter("CMS_jec", "lnN")
    massScale = rl.NuisanceParameter("CMS_msdScale", "shape")
    lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)

    ptbins = np.array([450, 600]) #, 675, 800, 1200])
    npt = len(ptbins) - 1
    msdbins = np.array([ 40.,  44.,  48.,  52.,  56.,  60.,  64.,  68.,  72.,  76.,  80.,
        84.,  88.,  92.,  96., 100., 104., 108., 112., 116., 120., 124.,
       128., 132., 136., 140., 144., 148., 152., 156., 160., 164., 168.,
       172., 176., 180., 184., 188., 192., 196., 200.])
    msd = rl.Observable("msd", msdbins)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij")
    rhopts = 2 * np.log(msdpts / ptpts)
    #below changed from 1200.0 - 450.0
    ptscaled = (ptpts - 450.0) / (600.0 - 450.0)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    # Build qcd MC pass+fail model and fit to polynomial
    qcdmodel = rl.Model("qcdmodel")
    qcdpass, qcdfail = 0.0, 0.0
    for ptbin in range(npt):
        failCh = rl.Channel("ptbin%d%s" % (ptbin, "fail"))
        passCh = rl.Channel("ptbin%d%s" % (ptbin, "pass"))
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)

        failTempl = load_hist('./hbb_vs_qcd_fail_wp_neg3.pkl')
        passTempl = load_hist('./hbb_vs_qcd_pass_wp_neg3.pkl')

        failCh.setObservation(failTempl)
        passCh.setObservation(passTempl)
        qcdfail += failCh.getObservation().sum()
        qcdpass += passCh.getObservation().sum()

    qcdeff = qcdpass / qcdfail
    tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (0, 0), ["pt", "rho"], limits=(0, 10))
    tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)
    for ptbin in range(npt):
        failCh = qcdmodel["ptbin%dfail" % ptbin]
        passCh = qcdmodel["ptbin%dpass" % ptbin]
        failObs = failCh.getObservation()
        qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
        sigmascale = 10.0
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
        fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
        passCh.addSample(pass_qcd)

        failCh.mask = validbins[ptbin]
        passCh.mask = validbins[ptbin]

    qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
    simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
    qcdfit = simpdf.fitTo(
        obs,
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.Save(),
        ROOT.RooFit.Minimizer("Minuit2", "migrad"),
        ROOT.RooFit.PrintLevel(-1),
    )
    qcdfit_ws.add(qcdfit)
    if "pytest" not in sys.modules:
        qcdfit_ws.writeToFile(os.path.join(str(tmpdir), "testModel_qcdfit.root"))
    if qcdfit.status() != 0:
        qcdfit.Print('v')
        raise RuntimeError("Could not fit qcd")

    param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
    decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + "_deco", qcdfit, param_names)
    tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
    tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
    tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", (0, 0), ["pt", "rho"], limits=(0, 10))
    tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
    tf_params = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params

    # build actual fit model now
    model = rl.Model("testModel")

    for ptbin in range(npt):
        for region in ["pass", "fail"]:
            ch = rl.Channel("ptbin%d%s" % (ptbin, region))
            model.addChannel(ch)

            isPass = region == "pass"
            ptnorm = 1.0
            templates = {
                "wqq": load_hist(f'./hbb_vs_wqq_{region}_wp_neg3.pkl'),
                "zqq": load_hist(f'./hbb_vs_zqq_{region}_wp_neg3.pkl'),
                "singletop": load_hist(f'./hbb_vs_singletop_{region}_wp_neg3.pkl'),
                "hbb": load_hist(f'./hbb_{region}_wp_neg3.pkl'),
                "ttboosted": load_hist(f'./hbb_vs_ttboosted_{region}_wp_neg3.pkl'),
                "ww": load_hist(f'./hbb_vs_ww_{region}_wp_neg3.pkl'),
                "wz": load_hist(f'./hbb_vs_wz_{region}_wp_neg3.pkl'),
                "zz": load_hist(f'./hbb_vs_zz_{region}_wp_neg3.pkl'),
            }
            for sName in ["zqq", "wqq", "singletop", "hbb", "ttboosted", "ww", "wz", "zz"]: 
                # some mock expectations
                templ = templates[sName]
                stype = rl.Sample.SIGNAL if sName == "hbb" else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)

                # mock systematics
               # jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
               # msdUp = np.linspace(0.9, 1.1, msd.nbins)
               # msdDn = np.linspace(1.2, 0.8, msd.nbins)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
               # sample.setParamEffect(jec, jecup_ratio)
               # sample.setParamEffect(massScale, msdUp, msdDn)
               # sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            # make up a data_obs, with possibly different yield values

            yields = sum(tpl.values() for tpl in templates.values())
            yields += load_hist(f'./hbb_vs_qcd_{region}_wp_neg3.pkl').values()
            if throwPoisson:
                yields = np.random.poisson(yields)
            data_obs = (yields, msd.binning, msd.name)
            ch.setObservation(data_obs)

            # drop bins outside rho validity
            mask = validbins[ptbin]
            # blind bins 11, 12, 13
            # mask[11:14] = False
            ch.mask = mask

    for ptbin in range(npt):
        failCh = model["ptbin%dfail" % ptbin]
        passCh = model["ptbin%dpass" % ptbin]

        qcdparams = np.array([rl.IndependentParameter("qcdparam_ptbin%d_msdbin%d" % (ptbin, i), 0) for i in range(msd.nbins)])
        initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
        for sample in failCh:
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)
        sigmascale = 10  # to scale the deviation from initial
        scaledparams = initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
        fail_qcd = rl.ParametericSample("ptbin%dfail_qcd" % ptbin, rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample("ptbin%dpass_qcd" % ptbin, rl.Sample.BACKGROUND, tf_params[ptbin, :], fail_qcd)
        passCh.addSample(pass_qcd)


    # Fill in muon CR
#    for region in ["pass", "fail"]:
#        ch = rl.Channel("muonCR%s" % (region,))
#        model.addChannel(ch)
#
#        isPass = region == "pass"
#        templates = {
#            "tqq": gaus_sample(norm=10 * (30 if isPass else 60), loc=150, scale=20, obs=msd),
#            "qcd": expo_sample(norm=10 * (5e2 if isPass else 1e3), scale=40, obs=msd),
#        }
#        for sName, templ in templates.items():
#            stype = rl.Sample.BACKGROUND
#            sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)
#
#            # mock systematics
#            jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
#            sample.setParamEffect(jec, jecup_ratio)
#
#            ch.addSample(sample)
#
#        # make up a data_obs
#        templates = {
#            "tqq": gaus_sample(norm=10 * (30 if isPass else 60), loc=150, scale=20, obs=msd),
#            "qcd": expo_sample(norm=10 * (5e2 if isPass else 1e3), scale=40, obs=msd),
#        }
#        yields = sum(tpl[0] for tpl in templates.values())
#        if throwPoisson:
#            yields = np.random.poisson(yields)
#        data_obs = (yields, msd.binning, msd.name)
#        ch.setObservation(data_obs)

#    tqqpass = model["muonCRpass_tqq"]
#    tqqfail = model["muonCRfail_tqq"]
#    tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
#    tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
#    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
#    tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
#    tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    with open(os.path.join(str(tmpdir), "testModel.pkl"), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), "testModel"))



if __name__ == "__main__":
    if not os.path.exists("hbb_out"):
        os.mkdir("hbb_out")
    test_rhalphabet("hbb_out")
