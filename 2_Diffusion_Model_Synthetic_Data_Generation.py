"""
Denoising Diffusion Probabilistic Model (DDPM) - synthetic EEG biometric data for 2,500 new individuals.

Data : PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB, 109 subjects),
       resting-state eyes-open (R01) / eyes-closed (R02), 320 log band-power
       features (64 channels x 5 bands) per 2 s window.
       (replaces the non-public BED dataset of the original project)
Model: 200-step cosine-schedule DDPM, MLP denoiser conditioned on identity code + state

Run  : python "2_Diffusion_Model_Synthetic_Data_Generation.py" [--n-identities 2500] [--epochs-per-state 20] [--fit-all-subjects]
Out  : Outputs/Diffusion/  (synthetic CSV/NPZ, metrics JSON, SHAP + overview plots)
"""
from eegsyn.pipeline import run

if __name__ == "__main__":
    run("Diffusion", default_epochs=300)
