"""
Gaussian Copula - synthetic EEG biometric data for 2,500 new individuals.

Data : PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB, 109 subjects),
       resting-state eyes-open (R01) / eyes-closed (R02), 320 log band-power
       features (64 channels x 5 bands) per 2 s window.
       (replaces the non-public BED dataset of the original project)
Model: Gaussian copula over [identity code, residual] with exact conditional sampling

Run  : python "4_Copula_Gaussian_Synthetic_Data_Generation.py" [--n-identities 2500] [--epochs-per-state 20] [--fit-all-subjects]
Out  : Outputs/Copula/  (synthetic CSV/NPZ, metrics JSON, SHAP + overview plots)
"""
from eegsyn.pipeline import run

if __name__ == "__main__":
    run("Copula", default_epochs=1)
