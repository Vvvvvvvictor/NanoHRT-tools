from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import numpy as np

from ..helpers.utils import deltaPhi, polarP4, deltaR
from ..helpers.triggerHelper import passTrigger
from ..helpers.muonSFHelper import MuonSFHelper
from .HeavyFlavBaseProducer import HeavyFlavBaseProducer


class ZmmSampleProducer(HeavyFlavBaseProducer):

    def __init__(self, **kwargs):
        super(ZmmSampleProducer, self).__init__(channel='zmm', **kwargs)

        # Initialize Muon SF Helpers for HLT, ID, and ISO
        # HLT: eta in [-2.4, 2.4), pt >= 52.0, NO abs(eta)
        self.muonHLTSFHelper = MuonSFHelper(
            year=self.year,
            sf_type='NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdTrkHighPt_and_TkIsoLoose',
            json_filename_pattern='muon_Z_{year}.json.gz',
            eta_range=(-2.4, 2.4),
            pt_range=(52.0, np.inf),
            use_abs_eta=False
        )
        # ID: eta in [-2.4, 2.4), pt >= 10.0, NO abs(eta)
        self.muonIDSFHelper = MuonSFHelper(
            year=self.year,
            sf_type='NUM_HighPtID_DEN_TrackerMuons',
            json_filename_pattern='muon_Z_{year}.json.gz',
            eta_range=(-2.4, 2.4),
            pt_range=(10.0, np.inf),
            use_abs_eta=False
        )
        # ISO: eta in [-2.4, 2.4), pt >= 10.0, NO abs(eta)
        self.muonISOSFHelper = MuonSFHelper(
            year=self.year,
            sf_type='NUM_LooseRelTkIso_DEN_HighPtID',
            json_filename_pattern='muon_Z_{year}.json.gz',
            eta_range=(-2.4, 2.4),
            pt_range=(10.0, np.inf),
            use_abs_eta=False
        )

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        super(ZmmSampleProducer, self).beginFile(inputFile, outputFile, inputTree, wrappedOutputTree)

        # Initialize all SF helpers
        self.muonHLTSFHelper.beginJob()
        self.muonIDSFHelper.beginJob()
        self.muonISOSFHelper.beginJob()

        # trigger variables
        self.out.branch("passMuTrig", "O")

        # event variables
        self.out.branch("leptonicZ_pt", "F")
        self.out.branch("leptonicZ_mass", "F")

        # Muon SF branches (HLT, ID, ISO)
        if self.isMC:
            for sf_name in ['muonHLTSF', 'muonIDSF', 'muonISOSF']:
                self.out.branch(sf_name, "F")
                self.out.branch(sf_name + "_stat_up", "F")
                self.out.branch(sf_name + "_stat_down", "F")
                self.out.branch(sf_name + "_syst_up", "F")
                self.out.branch(sf_name + "_syst_down", "F")
            # Combined ID*ISO SF
            self.out.branch("muonIDISOSF", "F")
            self.out.branch("muonIDISOSF_stat_up", "F")
            self.out.branch("muonIDISOSF_stat_down", "F")
            self.out.branch("muonIDISOSF_syst_up", "F")
            self.out.branch("muonIDISOSF_syst_down", "F")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        # muon selection
        self.correctMuons(event)
        event.muons = [mu for mu in event._allMuons if ((mu.pt > 15 and mu.looseId) or (mu.pt > 30 and mu.highPtId)) and abs(mu.eta) < 2.4 and mu.pfIsoId>1] # and abs(mu.dxy) < 0.2 and abs(mu.dz) < 0.5]
        if len(event.muons) != 2:
            return False
        selected_muons = sorted(event.muons, key=lambda mu: mu.pt, reverse=True)
        if selected_muons[0].pt < 60 or selected_muons[1].pt < 30:
            return False
        if event.muons[0].charge * event.muons[1].charge > 0:
            return False

        # leptonic Z pt and mass cut
        event.leptonicZ = polarP4(event.muons[0]) + polarP4(event.muons[1])
        if event.leptonicZ.Pt() < 450 or event.leptonicZ.M() < 70 or event.leptonicZ.M() > 100:
            return False

        self.selectLeptons(event)
        self.correctJetsAndMET(event)

        probe_jets = [fj for fj in event.fatjets if (deltaR(fj, event.muons[0]) > 0.8 and deltaR(fj, event.muons[1]) > 0.8)]
        if len(probe_jets) == 0:
            return False

        # # met selection
        # if event.met.pt < 50.0:
        #     return False

        # # at least one b-jet, in the same hemisphere of the muon
        # event.bjets = [j for j in event.ak4jets if j.btagDeepFlavB > self.DeepJet_WP_M and
        #                abs(deltaPhi(j, event.mu)) < 2]
        # if len(event.bjets) == 0:
        #     return False

        # # require fatjet away from the muon
        # probe_jets = [fj for fj in event.fatjets if abs(deltaPhi(fj, event.mu)) > 2]
        # if len(probe_jets) == 0:
        #     return False

        probe_jets = probe_jets[:1]
        self.loadGenHistory(event, probe_jets)
        self.evalTagger(event, probe_jets)
        self.evalMassRegression(event, probe_jets)

        # fill output branches
        self.fillBaseEventInfo(event)
        self.fillFatJetInfo(event, probe_jets)

        # fill
        self.out.fillBranch("passMuTrig", passTrigger(event, ['HLT_Mu50', 'HLT_TkMu50']))
        for i in range(2):
            prefix = "muon%d_" % i
            mu = event.muons[i]
            self.out.fillBranch(prefix + "pt", mu.pt)
            self.out.fillBranch(prefix + "eta", mu.eta)
            self.out.fillBranch(prefix + "phi", mu.phi)
            self.out.fillBranch(prefix + "mass", mu.mass)
            self.out.fillBranch(prefix + "miniIso", mu.miniPFRelIso_all)
            if self._muonSysts['musr_extra_br']:
                self.out.fillBranch(prefix + "pt_scaleUp", mu.pt_scaleUp)
                self.out.fillBranch(prefix + "pt_scaleDown", mu.pt_scaleDown)
                self.out.fillBranch(prefix + "pt_resUp", mu.pt_resUp)
                self.out.fillBranch(prefix + "pt_resDown", mu.pt_resDown)
        self.out.fillBranch("leptonicZ_pt", event.leptonicZ.Pt())
        self.out.fillBranch("leptonicZ_mass", event.leptonicZ.M())

        # Calculate and fill Muon SF (HLT, ID, ISO)
        if self.isMC:
            # HLT Scale Factors
            muonHLTSF = self.muonHLTSFHelper.getSFWithUncertainty(event.muons)
            self.out.fillBranch("muonHLTSF", muonHLTSF['nominal'])
            self.out.fillBranch("muonHLTSF_stat_up", muonHLTSF['stat_up'])
            self.out.fillBranch("muonHLTSF_stat_down", muonHLTSF['stat_down'])
            self.out.fillBranch("muonHLTSF_syst_up", muonHLTSF['syst_up'])
            self.out.fillBranch("muonHLTSF_syst_down", muonHLTSF['syst_down'])

            # ID Scale Factors
            muonIDSF = self.muonIDSFHelper.getSFWithUncertainty(event.muons)
            self.out.fillBranch("muonIDSF", muonIDSF['nominal'])
            self.out.fillBranch("muonIDSF_stat_up", muonIDSF['stat_up'])
            self.out.fillBranch("muonIDSF_stat_down", muonIDSF['stat_down'])
            self.out.fillBranch("muonIDSF_syst_up", muonIDSF['syst_up'])
            self.out.fillBranch("muonIDSF_syst_down", muonIDSF['syst_down'])

            # ISO Scale Factors
            muonISOSF = self.muonISOSFHelper.getSFWithUncertainty(event.muons)
            self.out.fillBranch("muonISOSF", muonISOSF['nominal'])
            self.out.fillBranch("muonISOSF_stat_up", muonISOSF['stat_up'])
            self.out.fillBranch("muonISOSF_stat_down", muonISOSF['stat_down'])
            self.out.fillBranch("muonISOSF_syst_up", muonISOSF['syst_up'])
            self.out.fillBranch("muonISOSF_syst_down", muonISOSF['syst_down'])

            # Combined ID*ISO Scale Factors
            self.out.fillBranch("muonIDISOSF", muonIDSF['nominal'] * muonISOSF['nominal'])
            self.out.fillBranch("muonIDISOSF_stat_up", muonIDSF['stat_up'] * muonISOSF['stat_up'])
            self.out.fillBranch("muonIDISOSF_stat_down", muonIDSF['stat_down'] * muonISOSF['stat_down'])
            self.out.fillBranch("muonIDISOSF_syst_up", muonIDSF['syst_up'] * muonISOSF['syst_up'])
            self.out.fillBranch("muonIDISOSF_syst_down", muonIDSF['syst_down'] * muonISOSF['syst_down'])

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def MuonTree_2016(): return ZmmSampleProducer(year=2016)
def MuonTree_2017(): return ZmmSampleProducer(year=2017)
def MuonTree_2018(): return ZmmSampleProducer(year=2018)
def MuonTree_2024(): return ZmmSampleProducer(year=2024)
