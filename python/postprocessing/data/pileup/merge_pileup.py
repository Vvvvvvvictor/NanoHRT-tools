# This is for data.
# For MC: pileupCalc.py -i Cert_Collisions2024_378981_386951_Golden.json --inputLumiJSON pileup_JSON-2024BCDEFGHI.txt --calcMode true --minBiasXsec 69200 --maxPileupBin 100 --numPileupBins 100 mcPileupUL2024.root --pileupHistName pu_mc
import ROOT

file_nominal = "dataPileupHistogram-2024BCDEFGHI-69200ub.root"
file_up      = "dataPileupHistogram-2024BCDEFGHI-72400ub.root"
file_down    = "dataPileupHistogram-2024BCDEFGHI-66000ub.root"

output_path = "PileupHistogram-UL2024-100bins_withVar.root"

def get_hist(filename, new_name):
    f = ROOT.TFile.Open(filename)
    h = f.Get("pileup")
    h.SetName(new_name)
    h.SetDirectory(0)
    f.Close()
    return h

h_nom   = get_hist(file_nominal, "pileup")
h_plus  = get_hist(file_up,      "pileup_plus")
h_minus = get_hist(file_down,    "pileup_minus")

f_out = ROOT.TFile(output_path, "RECREATE")
h_nom.Write()
h_plus.Write()
h_minus.Write()

print(f"Created: {output_path}")
f_out.Close()