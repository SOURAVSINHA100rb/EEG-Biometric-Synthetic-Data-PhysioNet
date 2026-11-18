"""Collect Outputs/<model>/<model>_metrics.json into one comparison table (Markdown + CSV)."""
import glob
import json
import os

import pandas as pd

import config

rows = []
for f in sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*", "*_metrics.json"))):
    m = json.load(open(f))
    r = {"Model": m["model"], "Identities": m["n_identities"], "Windows": m["n_windows"]}
    r["Syn closed-set EER %"] = 100 * m["closed_set_synthetic"]["eer"]
    r["Syn rank-1 %"] = 100 * m["closed_set_synthetic"]["rank1"]
    if "tstr" in m:
        r["TSTR EER % (syn only)"] = 100 * m["tstr"]["synthetic_only"]["eer"]
        r["TSTR EER % (real only)"] = 100 * m["tstr"]["real_train"]["eer"]
        r["TSTR EER % (real+syn)"] = 100 * m["tstr"]["real+synthetic"]["eer"]
        r["Cross-state EER % (syn only)"] = 100 * m["tstr"]["synthetic_only"]["eer_cross_state"]
    fi = m["fidelity"]
    r["Discriminator AUC"] = fi["discriminator_auc"]
    r["Frechet dist"] = fi["frechet_syn_vs_train"]
    r["Between-person var (real/syn)"] = f"{fi['between_person_var_share_real']:.2f}/{fi['between_person_var_share_syn']:.2f}"
    if "state_task" in m:
        s = m["state_task"]
        r["EO/EC XGB % syn->real"] = 100 * s["xgb_acc_syn->real(TSTR)"]
        r["EO/EC XGB % combined"] = 100 * s["xgb_acc_combined->real"]
        if "shap_top10_similarity" in s:
            r["SHAP top-10 sim %"] = 100 * s["shap_top10_similarity"]
    r["Re-ID rate %"] = 100 * m["privacy"]["reid_rate_vs_real_train"]
    r["Fit+sample s"] = m["fit_seconds"] + m["sample_seconds"]
    rows.append(r)

if not rows:
    raise SystemExit("No metrics found - run the generator scripts first.")
df = pd.DataFrame(rows)
order = ["VAE", "GAN", "Diffusion", "LSTM", "Copula"]
df["_o"] = df.Model.map({k: i for i, k in enumerate(order)})
df = df.sort_values("_o").drop(columns="_o")
df.to_csv(os.path.join(config.OUTPUT_DIR, "Model_Comparison.csv"), index=False)
md = df.to_markdown(index=False, floatfmt=".2f") if hasattr(df, "to_markdown") else df.to_string(index=False)
open(os.path.join(config.OUTPUT_DIR, "Model_Comparison.md"), "w").write(md + "\n")
print(md)
