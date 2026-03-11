import os
import logging
import numpy as np
import math
import awkward as ak
from scipy.special import erfinv, erf
import correctionlib

# Configure logger
logger = logging.getLogger('MuonCorrector')
logger.setLevel(logging.INFO)

class SeedSequence:
    """
    Helper class for deterministic random seed generation.
    """
    def __init__(self, seeds):
        # Ensure input seeds are converted to standard Python ints to prevent 
        # numpy types from propagating and causing overflow warnings later.
        self.seeds = [int(s) & 0xFFFFFFFF for s in seeds]

    def generate(self, n):
        if n <= 0:
            return []
        mult = 0x9e3779b9
        mix_const = 0x85ebca6b
        buffer = [0x8b8b8b8b] * n
        s = len(self.seeds)

        for i in range(min(n, s)):
            buffer[i] ^= (self.seeds[i] + mult * i) & 0xFFFFFFFF
        for i in range(s, n):
            buffer[i] ^= (mult * i) & 0xFFFFFFFF
        for k in range(n):
            z = (buffer[(k + n - 1) % n] ^ (buffer[k] >> 27)) & 0xFFFFFFFF
            
            # FIX: Explicitly cast 'z' to 'int()' before multiplication.
            # This forces Python's arbitrary-precision arithmetic instead of 
            # fixed-width NumPy scalar arithmetic, preventing the "overflow encountered" warning.
            buffer[k] = ((int(z) * mix_const) & 0xFFFFFFFF) ^ ((buffer[k] << 13) & 0xFFFFFFFF)
            
        return [x & 0xFFFFFFFF for x in buffer]

class CrystallBall:
    """
    Vectorized Crystal Ball distribution for resolution smearing.
    """
    def __init__(self, m, s, a, n):
        self.pi = 3.14159
        self.sqrtPiOver2 = np.sqrt(self.pi/2.0)
        self.sqrt2 = np.sqrt(2.0)
        self.m = ak.Array(m)
        self.s = ak.Array(s)
        self.a = ak.Array(a)
        self.n = ak.Array(n)
        self.fa = abs(self.a)
        self.ex = np.exp(-self.fa * self.fa/2)
        self.A  = (self.n/self.fa)**self.n * self.ex
        self.C1 = self.n/self.fa/(self.n-1) * self.ex
        self.D1 = 2 * self.sqrtPiOver2 * erf(self.fa/self.sqrt2)
        self.B = self.n/self.fa - self.fa
        self.C = (self.D1 + 2 * self.C1)/self.C1
        self.D = (self.D1 + 2 * self.C1)/2
        self.N = 1.0/self.s/(self.D1 + 2 * self.C1)
        self.k = 1.0/(self.n - 1)
        self.NA = self.N * self.A
        self.Ns = self.N * self.s
        self.NC = self.Ns * self.C1
        self.F = 1 - self.fa * self.fa/self.n
        self.G = self.s * self.n/self.fa
        self.cdfMa = self.cdf(self.m - self.a * self.s)
        self.cdfPa = self.cdf(self.m + self.a * self.s)

    def cdf(self, x):
        x = ak.Array(x)
        d = (x - self.m)/self.s
        result = ak.full_like(d, 1.0)
        c1a = (d < -self.a) & (self.F - self.s * d / self.G > 0)
        c1b = (d < -self.a) & (self.F - self.s * d / self.G <= 0)
        c2a = (d > self.a) & (self.F + self.s * d / self.G > 0)
        c2b = (d > self.a) & (self.F + self.s * d / self.G <= 0)
        c3 = ~c1a & ~c1b & ~c2a & ~c2b
        
        result = ak.where(c1a, self.NC / np.power(self.F - self.s * d / self.G, self.n - 1), result)
        result = ak.where(c1b, self.NC, result)
        result = ak.where(c2a, self.NC * (self.C - np.power(self.F + self.s * d / self.G, 1 - self.n)), result)
        result = ak.where(c2b, self.NC * self.C, result)
        result = ak.where(c3, self.Ns * (self.D - self.sqrtPiOver2 * erf(-d / self.sqrt2)), result)
        return result

    def invcdf(self, u):
        u = ak.Array(u)
        result = ak.zeros_like(u)
        c1a = (u < self.cdfMa) & (self.NC/u > 0)
        c1b = (u < self.cdfMa) & (self.NC/u <= 0)
        c2a = (u > self.cdfPa) &  (self.C - u/self.NC > 0)
        c2b = (u > self.cdfPa) &  (self.C - u/self.NC <= 0)
        c3 = ~c1a & ~c1b & ~c2a & ~c2b

        result = ak.where(c1a, self.m + self.G * (self.F - (self.NC / u) ** self.k), result)
        result = ak.where(c1b, self.m + self.G * self.F, result)
        result = ak.where(c2a, self.m - self.G * (self.F - (self.C - u / self.NC) ** (-self.k)), result)
        result = ak.where(c2b, self.m - self.G * self.F, result)
        result = ak.where(c3, self.m - self.sqrt2 * self.s * erfinv((self.D - u / self.Ns) / self.sqrtPiOver2), result)
        return result

class MuonCorrector(object):
    """
    Muon Scale and Resolution Corrector wrapping correctionlib.
    """

    def __init__(self, year, applySmearing=True, musr_extra_br=False):
        """
        musr_extra_br: If True, calculate and store up/down variations for Scale and Resolution.
                       New attributes will be added to muons: 
                       pt_scaleUp, pt_scaleDown, pt_resUp, pt_resDown
        """
        self.year = str(year)
        self.applySmearing = applySmearing
        self.musr_extra_br = musr_extra_br
        self.cset = None
        
        # Map years to specific JSON files.
        self.json_map = {
            "2016APV": "muon_Z.json.gz",
            "2016":    "muon_Z.json.gz",
            "2017":    "muon_Z.json.gz",
            "2018":    "muon_Z.json.gz",
            "2022":    "muon_Z_2022.json.gz",
            "2022EE":  "muon_Z_2022EE.json.gz",
            "2023":    "muon_Z_2023.json.gz",
            "2023BPix":"muon_Z_2023BPix.json.gz",
            "2024":    "muon_scalesmearing_24.json.gz", 
        }

        if self.year not in self.json_map:
             logger.warning(f"Year {self.year} not explicitly defined in MuonCorrector.")

    def beginJob(self):
        filename = self.json_map.get(self.year, None)
        if filename is None:
            raise RuntimeError(f"No Muon JSON defined for year {self.year}")

        base_path = os.environ.get('CMSSW_BASE', '.')
        full_path = os.path.join(base_path, 'src/PhysicsTools/NanoHRTTools/data/muon_corrections', filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Muon correction file not found at: {full_path}")

        logger.info(f"Loading Muon corrections from: {full_path}")
        
        if filename.endswith(".gz"):
            import gzip
            with gzip.open(full_path, 'rb') as f:
                self.cset = correctionlib.CorrectionSet.from_string(f.read().decode('utf-8'))
        else:
            self.cset = correctionlib.CorrectionSet.from_file(full_path)

    def _get_random_numbers(self, eta, phi, nL, event):
        phi_int = (((np.asarray(phi, dtype=np.float64) / math.pi) * (1 << 31 - 1)).astype(np.int64) & 0xFFF).astype(np.uint32)
        seeds_input = []
        for p in phi_int:
            seed_val = (int(event.event) & 0xFFFFF) + (int(event.luminosityBlock) << 20) + int(p)
            seeds_input.append(np.uint32(seed_val))

        seed_seq = SeedSequence(seeds_input)
        raw_ints = seed_seq.generate(len(eta))
        rng = np.random.Generator(np.random.PCG64(raw_ints))
        return rng.random(len(eta))

    def correctMuons(self, muons, event):
        if not muons:
            return

        # --- Vectorize Inputs ---
        pt = np.array([m.pt for m in muons])
        eta = np.array([m.eta for m in muons])
        phi = np.array([m.phi for m in muons])
        charge = np.array([m.charge for m in muons])
        nL = np.array([m.nTrackerLayers for m in muons])
        
        # --- 1. Scale Correction (Data & MC) ---
        dtmc = "mc" if self.musr_extra_br else "data"
        
        try:
            a = self.cset["a_" + dtmc].evaluate(eta, phi, "nom")
            m = self.cset["m_" + dtmc].evaluate(eta, phi, "nom")
        except Exception as e:
            logger.error("Failed to evaluate Scale corrections.")
            raise e

        # pt_corr = 1 / (m/pt + charge * a)
        denom = (m / pt) + (charge * a)
        pt_scale_corr = np.where(denom != 0, 1.0 / denom, pt)
        pt_scale_corr = np.where(pt_scale_corr < 0, pt, pt_scale_corr)

        # Store the intermediate result (scaled but not smeared)
        pt_scaled_only = pt_scale_corr.copy()
        
        # Final nominal PT container
        final_pt = pt_scale_corr

        # --- 2. Resolution Smearing (MC Only) ---
        
        # Prepare arrays for systematics calculation later
        k_nom = np.zeros_like(pt)
        sigma_pt = np.zeros_like(pt)
        rnd_cb_val = np.zeros_like(pt) # The random value (std * rndm)
        is_smearing_applied = False

        if self.applySmearing:
            is_smearing_applied = True
            try:
                cb_mean = self.cset["cb_params"].evaluate(np.abs(eta), nL, 0)
                cb_sigma = self.cset["cb_params"].evaluate(np.abs(eta), nL, 1)
                cb_n = self.cset["cb_params"].evaluate(np.abs(eta), nL, 2)
                cb_alpha = self.cset["cb_params"].evaluate(np.abs(eta), nL, 3)
            except Exception as e:
                logger.error("Failed to evaluate Smearing parameters.")
                raise e

            rnd_uniform = self._get_random_numbers(eta, phi, nL, event)
            cb = CrystallBall(cb_mean, cb_sigma, cb_alpha, cb_n)
            rnd_smear = cb.invcdf(rnd_uniform)

            p0 = self.cset["poly_params"].evaluate(np.abs(eta), nL, 0)
            p1 = self.cset["poly_params"].evaluate(np.abs(eta), nL, 1)
            p2 = self.cset["poly_params"].evaluate(np.abs(eta), nL, 2)
            
            sigma_pt = p0 + p1 * final_pt + p2 * final_pt * final_pt
            sigma_pt = np.maximum(0, sigma_pt)

            k_data = self.cset["k_data"].evaluate(np.abs(eta), "nom")
            k_mc = self.cset["k_mc"].evaluate(np.abs(eta), "nom")
            
            mask_res = k_mc < k_data
            k_nom[mask_res] = np.sqrt(k_data[mask_res]**2 - k_mc[mask_res]**2)
            
            # pt_smeared = pt * (1 + k * sigma * rand_CB)
            smear_term = (k_nom * sigma_pt * rnd_smear)
            smear_factor = 1.0 + smear_term
            final_pt = final_pt * smear_factor

            # Store random term for systematics reuse
            # logic from MuonScaRe: std_x_cb = (pt_wresol / pt_woresol - 1) / k_f
            # This is essentially (sigma_pt * rnd_smear) if k_nom > 0
            # We calculate it explicitly to handle k=0 cases for systematics
            pass

        # --- 3. Systematics Calculation (MC Only) ---
        pt_scale_up = final_pt
        pt_scale_dn = final_pt
        pt_res_up   = final_pt
        pt_res_dn   = final_pt

        if self.musr_extra_br:
            # -- A. Scale Uncertainty --
            # Formula from MuonScaRe: unc = pt^2 * sqrt( (m_stat/pt)^2 + a_stat^2 + ... )
            stat_a = self.cset["a_mc"].evaluate(eta, phi, "stat")
            stat_m = self.cset["m_mc"].evaluate(eta, phi, "stat")
            stat_rho = self.cset["m_mc"].evaluate(eta, phi, "rho_stat")

            # Apply uncertainty relative to the *final nominal pt* 
            # (MuonScaRe logic implies additive unc on raw, but we are post-processing.
            #  Approximation: apply the delta calculated at the nominal PT)
            
            term1 = (stat_m / final_pt)
            term2 = stat_a
            # 2 * charge * rho * (m/pt) * a
            term3 = 2 * charge * stat_rho * term1 * term2
            
            # Unc in GeV
            unc_scale = (final_pt**2) * np.sqrt(term1**2 + term2**2 + term3)
            
            pt_scale_up = final_pt + unc_scale
            pt_scale_dn = final_pt - unc_scale

            # -- B. Resolution Uncertainty --
            if is_smearing_applied:
                # Need k_mc stat unc
                k_unc = self.cset["k_mc"].evaluate(np.abs(eta), "stat")
                k_nom_mc = self.cset["k_mc"].evaluate(np.abs(eta), "nom") # The MC intrinsic resolution k factor
                
                # Logic from pt_resol_var in MuonScaRe.py:
                # std_x_cb = (pt_wresol / pt_woresol - 1) / k_mc_nom
                # NOTE: This assumes 'k_nom' used in step 2 was based on k_mc_nom.
                # Actually step 2 used k_extra = sqrt(k_data^2 - k_mc^2).
                # The systematics logic in MuonScaRe seems to imply varying the *smearing width*.
                
                # Let's follow MuonScaRe's `pt_resol_var` exactly:
                # It takes pt_woresol (scaled) and pt_wresol (smeared).
                # It uses k_mc("nom") as denominator.
                
                # Guard against divide by zero
                mask_k = k_nom_mc > 0
                std_x_cb = np.zeros_like(final_pt)
                # (Smeared / Scaled - 1) / k_mc_nom
                # Note: This recovers the "sigma * rnd" term scaled by relative k factors
                std_x_cb[mask_k] = (final_pt[mask_k] / pt_scaled_only[mask_k] - 1.0) / k_nom_mc[mask_k]

                # Up variation: k -> k + k_unc
                factor_up = 1.0 + (k_nom_mc + k_unc) * std_x_cb
                pt_res_up = pt_scaled_only * factor_up
                
                # Down variation: k -> k - k_unc
                factor_dn = 1.0 + (k_nom_mc - k_unc) * std_x_cb
                pt_res_dn = pt_scaled_only * factor_dn

                # Sanity filters (same as nominal)
                for arr in [pt_res_up, pt_res_dn]:
                    ratio = np.divide(arr, pt_scaled_only, out=np.ones_like(arr), where=pt_scaled_only!=0)
                    mask_bad = (ratio > 2.0) | (ratio < 0.1) | (arr < 0)
                    arr[mask_bad] = pt_scaled_only[mask_bad]

        # --- 4. Write back to objects ---
        for i, muon in enumerate(muons):
            old_pt = muon.pt
            new_pt = final_pt[i]
            
            # Sanity checks for nominal
            if np.isnan(new_pt) or (new_pt > 200 and old_pt <= 200): 
                new_pt = old_pt
            ratio = new_pt / old_pt if old_pt > 0 else 1.0
            if ratio > 2.0 or ratio < 0.1:
                new_pt = old_pt

            # Update Nominal
            muon.pt = new_pt

            # Store Systematics if requested
            if self.musr_extra_br:
                muon.pt_scaleUp   = pt_scale_up[i]
                muon.pt_scaleDown = pt_scale_dn[i]
                muon.pt_resUp     = pt_res_up[i]
                muon.pt_resDown   = pt_res_dn[i]

    def endJob(self):
        self.cset = None

class MuonSFCalculator(object):
    """
    Muon Trigger/ID Scale Factor Calculator wrapping correctionlib.
    """

    def __init__(self, year):
        self.year = str(year)
        self.cset = None
        
        # Map years to specific JSON files.
        self.trigger_json_map = {
            "2016APV": "muon_trigger_SF_2016.json.gz",
            "2016":    "muon_trigger_SF_2016.json.gz",
            "2017":    "muon_trigger_SF_2017.json.gz",
            "2018":    "muon_trigger_SF_2018.json.gz",
            "2022":    "muon_trigger_SF_2022.json.gz",
            "2022EE":  "muon_trigger_SF_2022EE.json.gz",
            "2023":    "muon_trigger_SF_2023.json.gz",
            "2023BPix":"muon_trigger_SF_2023BPix.json.gz",
            "2024":    "muon_trigger_SF_2024.json.gz",
        }

        self.id_json_map = {
            "2016APV": "muon_ID_SF_2016.json.gz",
            "2016":    "muon_ID_SF_2016.json.gz",
            "2017":    "muon_ID_SF_2017.json.gz",
            "2018":    "muon_ID_SF_2018.json.gz",
            "2022":    "muon_ID_SF_2022.json.gz",
            "2022EE":  "muon_ID_SF_2022EE.json.gz",
            "2023":    "muon_ID_SF_2023.json.gz",
            "2023BPix":"muon_ID_SF_2023BPix.json.gz",
            "2024":    "muon_ID_SF_2024.json.gz",
        }

        if self.year not in self.trigger_json_map or self.year not in self.id_json_map:
             logger.warning(f"Year {self.year} not explicitly defined in MuonSFCalculator.")