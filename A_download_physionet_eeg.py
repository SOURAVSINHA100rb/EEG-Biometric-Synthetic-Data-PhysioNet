"""
Step A - Download the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).

Dataset : https://physionet.org/content/eegmmidb/1.0.0/
Licence : Open Data Commons Attribution License v1.0 (free, no registration)
Subjects: 109 volunteers, 64-channel EEG (BCI2000, 10-10 system), 160 Hz, EDF+

For biometric authentication the two 1-minute *resting-state baseline* runs are
the standard choice (used in most EEG-biometrics papers on this dataset):
    R01 = eyes open (EO)   R02 = eyes closed (EC)
Add more runs with --runs if you want more epochs per person (R03-R14 are
motor/imagery tasks, 2 min each, ~2.5 MB per file).

Usage:
    python A_download_physionet_eeg.py                  # R01+R02, all 109 subjects (~260 MB)
    python A_download_physionet_eeg.py --runs 1 2 3 7   # more runs
    python A_download_physionet_eeg.py --subjects 1 2 3 # quick test
"""
import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"


def edf_name(subject, run):
    return f"S{subject:03d}R{run:02d}.edf"


def download_one(subject, run, out_dir, retries=4):
    name = edf_name(subject, run)
    sub_dir = os.path.join(out_dir, f"S{subject:03d}")
    os.makedirs(sub_dir, exist_ok=True)
    dst = os.path.join(sub_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        return name, "exists"
    url = f"{BASE_URL}/S{subject:03d}/{name}"
    tmp = dst + ".part"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eeg-synthetic-data-downloader"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dst)
            return name, "ok"
        except Exception as e:  # network hiccup -> retry with back-off
            if attempt == retries:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return name, f"FAILED ({e})"
            time.sleep(2 * attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 110)))
    p.add_argument("--runs", type=int, nargs="*", default=list(config.RUNS.keys()))
    p.add_argument("--out", default=config.RAW_DIR)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    jobs = [(s, r) for s in a.subjects for r in a.runs]
    print(f"Downloading {len(jobs)} EDF files ({len(a.subjects)} subjects x runs {a.runs}) -> {a.out}")
    failed = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(download_one, s, r, a.out) for s, r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            if status.startswith("FAILED"):
                failed.append(name)
            if i % 20 == 0 or i == len(jobs) or status.startswith("FAILED"):
                print(f"  [{i}/{len(jobs)}] {name}: {status}", flush=True)
    if failed:
        print(f"\n{len(failed)} files failed: {failed}\nRe-run the script to retry.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
