"""Shared settings for the PhysioNet EEG biometric synthetic-data project."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "physionet_eegmmidb")
FEATURES_CSV = os.path.join(DATA_DIR, "PhysioNet_EEG_Biometric_Features.csv")
OUTPUT_DIR = os.path.join(ROOT, "Outputs")

# Runs used and the recording state they represent
RUNS = {1: "EO", 2: "EC"}          # R01 eyes-open rest, R02 eyes-closed rest
STATES = ["EO", "EC"]

# Signal processing
SFREQ = 160.0                       # dataset sampling rate (a few files are resampled to this)
L_FREQ, H_FREQ = 1.0, 45.0          # band-pass (also removes 60 Hz mains)
EPOCH_SEC = 2.0                     # window length
STEP_SEC = 1.0                      # hop (50 % overlap)
REJECT_UV = 800.0                   # drop windows whose peak-to-peak exceeds this (µV); gross artifacts only

BANDS = {                           # log10 band power features per channel
    "DELTA": (1, 4),
    "THETA": (4, 8),
    "ALPHA": (8, 13),
    "BETA": (13, 30),
    "GAMMA": (30, 45),
}

MIN_WINDOWS_PER_STATE = 10           # subjects with fewer clean windows in EO or EC are excluded

# Synthetic dataset
N_SYNTHETIC_IDENTITIES = 2500
EPOCHS_PER_STATE = 20               # synthetic windows per person per state (EO / EC)

# Evaluation split (generators are fitted on TRAIN subjects only, the
# remaining real subjects are never seen and are used to test utility/privacy)
TEST_SUBJECT_FRACTION = 0.30
SEED = 42
