import os
import logging
import numpy as np
import correctionlib

# Configure logger
logger = logging.getLogger('MuonSFHelper')
logger.setLevel(logging.INFO)

class MuonSFHelper(object):
    """
    Generic Muon Scale Factor Helper wrapping correctionlib.
    Can be used for ID, ISO, HLT, etc.
    """

    def __init__(self, year, sf_type, json_filename_pattern, eta_range=None, pt_range=None, use_abs_eta=True):
        """
        Args:
            year: Year string (e.g., '2016', '2017', '2018', '2024')
            sf_type: SF type name in the JSON file (e.g., 'NUM_HighPtID_DEN_GlobalMuonProbes')
            json_filename_pattern: Pattern for JSON filename (e.g., 'muon_ID_SF_{year}.json.gz')
            eta_range: Tuple (min, max) for valid eta range. If None, no clipping
            pt_range: Tuple (min, max) for valid pt range. If None, no clipping
            use_abs_eta: If True, use absolute value of eta (default True)
        """
        self.year = str(year)
        self.sf_type = sf_type
        self.json_filename_pattern = json_filename_pattern
        self.eta_range = eta_range
        self.pt_range = pt_range
        self.use_abs_eta = use_abs_eta
        self.cset = None

        if self.year not in ["2016APV", "2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix", "2024"]:
            logger.warning(f"Year {self.year} not explicitly defined in MuonSFHelper.")

    def beginJob(self):
        # Replace {year} placeholder in the filename pattern
        filename = self.json_filename_pattern.format(year=self.year)

        base_path = os.environ.get('CMSSW_BASE', '.')
        full_path = os.path.join(base_path, 'src/PhysicsTools/NanoHRTTools/data/muon_corrections', filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Muon SF file not found at: {full_path}")

        logger.info(f"Loading Muon SF ({self.sf_type}) from: {full_path}")

        if filename.endswith(".gz"):
            import gzip
            with gzip.open(full_path, 'rb') as f:
                self.cset = correctionlib.CorrectionSet.from_string(f.read().decode('utf-8'))
        else:
            self.cset = correctionlib.CorrectionSet.from_file(full_path)

    def getSF(self, muons, systematic='nominal'):
        """
        Calculate scale factors for a list of muons.

        Args:
            muons: List of muon objects with pt, eta attributes
            systematic: Systematic variation ('nominal', 'stat', 'syst', 'systup', 'systdown', etc.)

        Returns:
            numpy array of scale factors
        """
        if not muons:
            return np.array([])

        # Vectorize inputs
        pt = np.array([mu.pt for mu in muons])
        eta = np.array([mu.eta for mu in muons])

        # Apply eta transformation if needed
        if self.use_abs_eta:
            eta_input = np.abs(eta)
        else:
            eta_input = eta

        # Check if muons are within valid ranges
        valid_mask = np.ones(len(muons), dtype=bool)

        if self.eta_range is not None:
            eta_min, eta_max = self.eta_range
            # For eta check, use the original eta (not transformed)
            check_eta = np.abs(eta) if self.use_abs_eta else eta
            valid_mask &= (check_eta >= eta_min) & (check_eta < eta_max)

        if self.pt_range is not None:
            pt_min, pt_max = self.pt_range
            valid_mask &= (pt >= pt_min)
            if pt_max != np.inf:
                valid_mask &= (pt < pt_max)

        # Initialize SF array with 1.0 for all muons
        sf = np.ones(len(muons))

        # Only evaluate SF for valid muons
        if np.any(valid_mask):
            try:
                # Clip values to valid ranges for correctionlib evaluation
                pt_clipped = pt[valid_mask]
                eta_clipped = eta_input[valid_mask]

                if self.pt_range is not None:
                    pt_clipped = np.clip(pt_clipped, self.pt_range[0],
                                        self.pt_range[1] if self.pt_range[1] != np.inf else pt_clipped.max())

                if self.eta_range is not None:
                    # Clip to slightly inside the range to avoid boundary issues
                    eta_min_clip = self.eta_range[0] #+ 0.001
                    eta_max_clip = self.eta_range[1] #- 0.001
                    eta_clipped = np.clip(eta_clipped, eta_min_clip, eta_max_clip)

                sf_valid = np.array([
                    self.cset[self.sf_type].evaluate(float(e), float(p), systematic)
                    for e, p in zip(eta_clipped, pt_clipped)
                ])
                sf[valid_mask] = sf_valid

            except Exception as e:
                logger.error(f"Failed to evaluate Muon SF ({self.sf_type}): {e}")
                raise e

        return sf

    def getSFProduct(self, muons, systematic='nominal'):
        """
        Calculate the product of ID scale factors for all muons.

        Args:
            muons: List of muon objects
            systematic: Systematic variation

        Returns:
            float: Product of all scale factors
        """
        sf = self.getSF(muons, systematic)
        return np.prod(sf) if len(sf) > 0 else 1.0

    def getSFWithUncertainty(self, muons):
        """
        Calculate scale factors with uncertainties.

        Args:
            muons: List of muon objects

        Returns:
            dict: Dictionary containing 'nominal', 'stat_up', 'stat_down', 'syst_up', 'syst_down'
        """
        sf_nominal = self.getSF(muons, 'nominal')

        # Try to get stat uncertainty (might be returned as absolute uncertainty)
        try:
            sf_stat = self.getSF(muons, 'stat')
            # If 'stat' returns uncertainty value, calculate up/down
            # Assuming stat returns the SF itself with uncertainty already applied
            # or returns the uncertainty value
            sf_stat_up = sf_nominal + np.abs(sf_stat - sf_nominal)
            sf_stat_down = sf_nominal - np.abs(sf_stat - sf_nominal)
        except Exception:
            # If 'stat' doesn't exist, set to nominal
            sf_stat_up = sf_nominal
            sf_stat_down = sf_nominal

        # Get systematic uncertainties
        try:
            sf_syst_up = self.getSF(muons, 'systup')
        except Exception:
            sf_syst_up = sf_nominal

        try:
            sf_syst_down = self.getSF(muons, 'systdown')
        except Exception:
            sf_syst_down = sf_nominal

        return {
            'nominal': np.prod(sf_nominal) if len(sf_nominal) > 0 else 1.0,
            'stat_up': np.prod(sf_stat_up) if len(sf_stat_up) > 0 else 1.0,
            'stat_down': np.prod(sf_stat_down) if len(sf_stat_down) > 0 else 1.0,
            'syst_up': np.prod(sf_syst_up) if len(sf_syst_up) > 0 else 1.0,
            'syst_down': np.prod(sf_syst_down) if len(sf_syst_down) > 0 else 1.0,
        }

    def endJob(self):
        self.cset = None
