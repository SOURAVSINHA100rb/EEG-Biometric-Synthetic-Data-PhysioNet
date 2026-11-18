"""
Step B - Turn the raw PhysioNet EDF files into a biometric feature table.

Pipeline per recording (R01 eyes-open, R02 eyes-closed):
  1. read EDF, keep the 64 EEG channels, resample to 160 Hz if needed
  2. band-pass 1-45 Hz, common average reference
  3. cut into 2 s windows with 1 s hop, reject windows > 800 µV peak-to-peak
  4. Welch PSD per window/channel -> log10 band power in 5 bands
     (DELTA, THETA, ALPHA, BETA, GAMMA)  => 64 x 5 = 320 features

Output: data/PhysioNet_EEG_Biometric_Features.csv
  columns  Fc5_DELTA ... Iz_GAMMA | Subject | State | Run | Window
This replaces the "Physiological Emotion Recognition EEG Data.csv" (BED) file
of the original project; the label is now the person's identity (Subject).
"""
import argparse
import glob
import os

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

import config

mne.set_log_level("ERROR")


def clean_channel(name):
    return name.strip(".").strip()


def extract_recording(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels(clean_channel)
    raw.pick("eeg")
    if abs(raw.info["sfreq"] - config.SFREQ) > 1e-3:
        raw.resample(config.SFREQ)
    raw.filter(config.L_FREQ, config.H_FREQ, fir_design="firwin", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    data = raw.get_data() * 1e6                      # volts -> µV, shape (64, n)
    fs = raw.info["sfreq"]
    win, hop = int(config.EPOCH_SEC * fs), int(config.STEP_SEC * fs)
    starts = np.arange(0, data.shape[1] - win + 1, hop)
    epochs = np.stack([data[:, s:s + win] for s in starts])          # (E, 64, win)
    ptp = epochs.max(-1) - epochs.min(-1)
    keep = (ptp < config.REJECT_UV).all(1)
    epochs = epochs[keep]
    if len(epochs) == 0:
        return None, raw.ch_names, 0

    freqs, psd = welch(epochs, fs=fs, nperseg=int(fs), noverlap=int(fs) // 2, axis=-1)
    feats = []
    for lo, hi in config.BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        feats.append(np.log10(psd[..., m].mean(-1) + 1e-12))         # (E, 64)
    feats = np.stack(feats, -1).reshape(len(epochs), -1)            # channel-major: ch0_b0..ch0_b4, ch1_b0..
    return feats.astype(np.float32), raw.ch_names, int((~keep).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=config.RAW_DIR)
    p.add_argument("--out", default=config.FEATURES_CSV)
    a = p.parse_args()

    rows, ch_names, n_rej = [], None, 0
    files = sorted(glob.glob(os.path.join(a.raw, "S*", "S*R*.edf")))
    if not files:
        raise SystemExit(f"No EDF files under {a.raw}. Run A_download_physionet_eeg.py first.")
    for i, f in enumerate(files, 1):
        base = os.path.basename(f)
        subj, run = int(base[1:4]), int(base[5:7])
        if run not in config.RUNS:
            continue
        feats, names, rej = extract_recording(f)
        n_rej += rej
        if feats is None:
            print(f"  {base}: all windows rejected, skipped")
            continue
        if ch_names is None:
            ch_names = names
        elif names != ch_names:
            raise RuntimeError(f"Channel order mismatch in {base}")
        df = pd.DataFrame(feats, columns=[f"{c}_{b}" for c in ch_names for b in config.BANDS])
        df["Subject"] = subj
        df["State"] = config.RUNS[run]
        df["Run"] = run
        df["Window"] = np.arange(len(df))
        rows.append(df)
        if i % 25 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files", flush=True)

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.5f")
    per = out.groupby("Subject").size()
    print(f"\nSaved {a.out}")
    print(f"  {len(out)} windows | {out.Subject.nunique()} subjects | {len(ch_names)} channels x "
          f"{len(config.BANDS)} bands = {len(ch_names) * len(config.BANDS)} features")
    print(f"  windows per subject: min {per.min()}, median {int(per.median())}, max {per.max()}  "
          f"(rejected {n_rej} noisy windows)")


if __name__ == "__main__":
    main()
