"""
Variational LSTM (sequence VAE over the 64 electrodes) - synthetic EEG biometric data for 2,500 new individuals.

Data : PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB, 109 subjects),
       resting-state eyes-open (R01) / eyes-closed (R02), 320 log band-power
       features (64 channels x 5 bands) per 2 s window.
       (replaces the non-public BED dataset of the original project)
Model: BiLSTM encoder / LSTM decoder conditioned on identity code + state

Run  : python "3_VLSTM_Synthetic_Data_Generation.py" [--n-identities 2500] [--epochs-per-state 20] [--fit-all-subjects]
Out  : Outputs/LSTM/  (synthetic CSV/NPZ, metrics JSON, SHAP + overview plots)
"""
from eegsyn.pipeline import run

if __name__ == "__main__":
    run("LSTM", default_epochs=30, batch_size=128)
