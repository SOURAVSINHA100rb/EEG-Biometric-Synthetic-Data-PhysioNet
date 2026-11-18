#!/usr/bin/env bash
# Full pipeline: download -> features -> 5 generators (2,500 identities each) -> comparison table
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
$PY A_download_physionet_eeg.py
$PY B_extract_biometric_features.py
for s in 0_VAE_Synthetic_Data_Generation.py 1_GAN_Synthetic_Data_Generation.py \
         2_Diffusion_Model_Synthetic_Data_Generation.py 3_VLSTM_Synthetic_Data_Generation.py \
         4_Copula_Gaussian_Synthetic_Data_Generation.py; do
  $PY "$s" "$@" 2>&1 | grep -v -i "warning" | tee "logs_${s%.py}.txt"
done
$PY 5_Compare_All_Models.py
