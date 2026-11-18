"""
End-to-end runner shared by the five generator scripts (0_VAE ... 4_Copula).

  load PhysioNet features -> split subjects (train / unseen test)
  -> standardise -> identity model -> fit generator on within-person residuals
  -> sample 2,500 new identities x (EO, EC) x EPOCHS_PER_STATE windows
  -> save CSV / NPZ -> evaluate (biometrics, TSTR, fidelity, privacy, SHAP)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from eegsyn import evaluation as ev
from eegsyn.generators import GENERATORS
from eegsyn.identity import IdentityModel

META_COLS = ["Subject", "State", "Run", "Window"]


def parse_args(model_name, default_epochs):
    p = argparse.ArgumentParser(description=f"{model_name} synthetic EEG biometric data generation")
    p.add_argument("--n-identities", type=int, default=config.N_SYNTHETIC_IDENTITIES)
    p.add_argument("--epochs-per-state", type=int, default=config.EPOCHS_PER_STATE,
                   help="synthetic windows per person per state (EO and EC)")
    p.add_argument("--train-epochs", type=int, default=default_epochs, help="training epochs of the generator")
    p.add_argument("--identity-dim", type=int, default=32)
    p.add_argument("--fit-all-subjects", action="store_true",
                   help="fit on all 109 subjects (final release). Disables the unseen-subject TSTR test.")
    p.add_argument("--no-csv", action="store_true", help="only write the compressed .npz")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def load_features(path=config.FEATURES_CSV):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run A_download_physionet_eeg.py and B_extract_biometric_features.py")
    df = pd.read_csv(path)
    n = df.groupby(["Subject", "State"]).size().unstack(fill_value=0)
    bad = n.index[(n < config.MIN_WINDOWS_PER_STATE).any(axis=1)]
    if len(bad):
        print(f"Excluding subjects with < {config.MIN_WINDOWS_PER_STATE} clean windows in a state: {list(bad)}")
        df = df[~df.Subject.isin(bad)].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def split_subjects(subjects, frac, seed):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(subjects))
    n_test = int(round(frac * len(ids)))
    return np.sort(ids[n_test:]), np.sort(ids[:n_test])


def run(model_name, default_epochs, **gen_kwargs):
    a = parse_args(model_name, default_epochs)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    out_dir = os.path.join(config.OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, model_name)

    # ------------------------------------------------------------ data
    df, feat_cols = load_features()
    subj_all, state_all = df.Subject.values, df.State.values
    if a.fit_all_subjects:
        train_ids, test_ids = np.unique(subj_all), np.array([], dtype=int)
    else:
        train_ids, test_ids = split_subjects(subj_all, config.TEST_SUBJECT_FRACTION, a.seed)
    tr, te = np.isin(subj_all, train_ids), np.isin(subj_all, test_ids)
    scaler = StandardScaler().fit(df.loc[tr, feat_cols].values)
    X = scaler.transform(df[feat_cols].values).astype(np.float32)
    Xtr, str_, sttr = X[tr], subj_all[tr], state_all[tr]
    print(f"\n=== {model_name} | PhysioNet EEGMMIDB | {len(feat_cols)} features | "
          f"{len(train_ids)} train subjects ({tr.sum()} windows), {len(test_ids)} unseen test subjects ===")

    # ------------------------------------------------- identity model
    idm = IdentityModel(a.identity_dim, seed=a.seed).fit(Xtr, str_, sttr, config.STATES)
    print(f"Identity model: {idm.k}-dim codes explain {idm.explained_variance() * 100:.1f}% of "
          f"between-person prototype variance")
    R, C = idm.residuals(Xtr, str_, sttr)
    S = (sttr == "EC").astype(np.int64)

    # ------------------------------------------------------ generator
    gen = GENERATORS[model_name](epochs=a.train_epochs, seed=a.seed, **gen_kwargs)
    t_fit = time.time()
    gen.fit(R, C, S)
    t_fit = time.time() - t_fit

    # ------------------------------------------- 2,500 new identities
    codes, protos = idm.sample_identities(a.n_identities)
    n_per = a.epochs_per_state
    rows_c = np.repeat(codes, 2 * n_per, axis=0)
    rows_s = np.tile(np.repeat([0, 1], n_per), a.n_identities)
    rows_id = np.repeat(np.arange(a.n_identities), 2 * n_per)
    t_gen = time.time()
    Rs = gen.sample(rows_c, rows_s).astype(np.float32)
    mu = np.where(rows_s[:, None] == 0, protos[rows_id, :idm.D], protos[rows_id, idm.D:])
    Xs = (mu + Rs).astype(np.float32)
    t_gen = time.time() - t_gen
    subj_s = np.array([f"SYN{i + 1:04d}" for i in range(a.n_identities)])[rows_id]
    state_s = np.array(config.STATES)[rows_s]
    order_s = np.tile(np.arange(n_per), 2 * a.n_identities)
    print(f"Generated {len(Xs)} windows = {a.n_identities} identities x 2 states x {n_per} "
          f"(fit {t_fit:.0f}s, sampling {t_gen:.0f}s)")

    # ---------------------------------------------------------- save
    X_raw = scaler.inverse_transform(Xs).astype(np.float32)     # back to log10 band-power units
    np.savez_compressed(f"{prefix}_Synthetic_EEG_{a.n_identities}_Identities.npz",
                        X=X_raw, subject=subj_s, state=state_s, feature_names=np.array(feat_cols),
                        identity_codes=codes)
    if not a.no_csv:
        out = pd.DataFrame(X_raw, columns=feat_cols)
        out["Subject"], out["State"] = subj_s, state_s
        out["Window"] = order_s
        csv = f"{prefix}_Synthetic_EEG_{a.n_identities}_Identities.csv"
        out.to_csv(csv, index=False, float_format="%.5f")
        print(f"Saved {csv}")

    if a.no_eval:
        return
    metrics = evaluate(model_name, prefix, X, subj_all, state_all, df.Window.values, tr, te, Xs, subj_s,
                       state_s, order_s, idm, codes, feat_cols, rng)
    metrics.update(model=model_name, n_identities=a.n_identities, n_windows=int(len(Xs)),
                   train_subjects=int(len(train_ids)), test_subjects=int(len(test_ids)),
                   fit_seconds=round(t_fit, 1), sample_seconds=round(t_gen, 1),
                   total_seconds=round(time.time() - t0, 1))
    with open(f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print_report(metrics)
    print(f"\nTotal runtime: {metrics['total_seconds']:.0f} s.  Outputs in {out_dir}")


def evaluate(model_name, prefix, X, subj, state, order, tr, te, Xs, subj_s, state_s, order_s, idm, codes,
             feat_cols, rng):
    print("\nEvaluating ...")
    m = {}
    Xtr, str_, sttr, otr = X[tr], subj[tr], state[tr], order[tr]
    has_test = te.any()

    # 1. biometric quality of the synthetic population vs real population
    #    closed set: LDA fitted on the first half of every recording, verification
    #    (enrol / probe, again split by time) on the second half
    def closed_set(Xa, sa, sta, oa):
        fit_m = ev.first_half(sa, sta, oa)
        emb = ev.LDAEmbedder(128).fit(Xa[fit_m], sa[fit_m])
        return ev.verify(emb, Xa[~fit_m], sa[~fit_m], sta[~fit_m], oa[~fit_m], rng=rng)

    v_real_closed = closed_set(Xtr, str_, sttr, otr)
    few = np.isin(subj_s, np.unique(subj_s)[: len(np.unique(str_))])     # same number of ids as real
    v_few = closed_set(Xs[few], subj_s[few], state_s[few], order_s[few])
    v_syn = closed_set(Xs, subj_s, state_s, order_s)
    for key, v in (("closed_set_real", v_real_closed), ("closed_set_synthetic_same_size", v_few),
                   ("closed_set_synthetic", v_syn)):
        m[key] = {"eer": v["eer"], "rank1": v["rank1"], "n_ids": v["n_ids"]}

    # 2. TSTR: embeddings trained on real-train / synthetic / both, tested on UNSEEN real people
    emb_r = ev.LDAEmbedder(128).fit(Xtr, str_)
    if has_test:
        Xte, ste, stte, ote = X[te], subj[te], state[te], order[te]
        emb_s = ev.LDAEmbedder(128).fit(Xs, subj_s)
        emb_c = ev.LDAEmbedder(128).fit(np.r_[Xtr, Xs], np.r_[str_.astype(str), subj_s])
        m["tstr"] = {}
        for tag, emb in (("real_train", emb_r), ("synthetic_only", emb_s), ("real+synthetic", emb_c)):
            r1 = ev.verify(emb, Xte, ste, stte, ote, rng=rng)
            r2 = ev.verify(emb, Xte, ste, stte, ote, protocol="cross_state", rng=rng)
            m["tstr"][tag] = {"eer": r1["eer"], "rank1": r1["rank1"],
                              "eer_cross_state": r2["eer"], "rank1_cross_state": r2["rank1"]}
            if tag == "real_train":
                v_real_test = r1
        test_ref = X[te]
    else:
        v_real_test, test_ref = v_real_closed, Xtr

    # 3. fidelity
    m["fidelity"] = ev.fidelity(Xtr, test_ref, Xs, str_, subj_s, rng)
    if has_test:
        m["state_task"] = ev.state_task(Xtr, (sttr == "EC").astype(int), X[te], (state[te] == "EC").astype(int),
                                        Xs, (state_s == "EC").astype(int), feat_cols, rng, prefix)

    # 4. privacy
    m["privacy"] = ev.privacy(idm, codes, emb_r, Xtr, str_, Xs, subj_s, rng)

    plot_overview(prefix, model_name, Xtr, sttr, Xs, state_s, idm, codes, feat_cols, v_real_test, v_syn)
    return m


def plot_overview(prefix, model_name, Xr, sr, Xs, ss, idm, codes, feat_cols, v_real, v_syn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = list(config.BANDS)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    # (a) band profile averaged over channels
    x = np.arange(len(bands))
    for st, ls in (("EO", "-"), ("EC", "--")):
        r = Xr[sr == st].reshape(-1, len(feat_cols) // len(bands), len(bands)).mean((0, 1))
        s = Xs[ss == st].reshape(-1, len(feat_cols) // len(bands), len(bands)).mean((0, 1))
        ax[0].plot(x, r, ls, color="#1f77b4", marker="o", label=f"real {st}")
        ax[0].plot(x, s, ls, color="#d62728", marker="s", label=f"synthetic {st}")
    ax[0].set_xticks(x, bands); ax[0].set_ylabel("standardised log power"); ax[0].legend(fontsize=8)
    ax[0].set_title("Mean band profile (all channels)")
    # (b) identity codes
    ax[1].scatter(codes[:, 0], codes[:, 1], s=4, alpha=0.3, color="#d62728", label=f"synthetic ({len(codes)})")
    ax[1].scatter(idm.codes[:, 0], idm.codes[:, 1], s=22, color="#1f77b4", edgecolor="k", label=f"real ({len(idm.codes)})")
    ax[1].set_xlabel("identity PC1"); ax[1].set_ylabel("identity PC2"); ax[1].legend(fontsize=8)
    ax[1].set_title("Identity space: real vs synthetic people")
    # (c) score distributions
    bins = np.linspace(-1, 1, 80)
    ax[2].hist(v_real["impostor"], bins, density=True, alpha=0.35, color="#1f77b4", label="real impostor")
    ax[2].hist(v_real["genuine"], bins, density=True, histtype="step", lw=2, color="#1f77b4",
               label=f"real genuine (EER {v_real['eer'] * 100:.1f}%)")
    ax[2].hist(v_syn["impostor"], bins, density=True, alpha=0.35, color="#d62728", label="synthetic impostor")
    ax[2].hist(v_syn["genuine"], bins, density=True, histtype="step", lw=2, color="#d62728",
               label=f"synthetic genuine (EER {v_syn['eer'] * 100:.1f}%)")
    ax[2].set_xlabel("cosine score"); ax[2].legend(fontsize=8); ax[2].set_title("Verification scores")
    fig.suptitle(f"{model_name}: synthetic EEG biometric data vs PhysioNet EEGMMIDB", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{prefix}_Overview.jpg", dpi=160)
    fig.savefig(f"{prefix}_Overview.pdf")
    plt.close(fig)


def print_report(m):
    pct = lambda v: f"{v * 100:6.2f}%"
    print("\n" + "=" * 72)
    print(f" {m['model']} - results")
    print("=" * 72)
    print(" Biometric quality (closed set; LDA on 1st half of each recording, enrol/probe on 2nd half)")
    for k, lab in (("closed_set_real", "real train subjects"), ("closed_set_synthetic_same_size", "synthetic (same #ids)"),
                   ("closed_set_synthetic", "synthetic (all ids)")):
        r = m[k]
        print(f"   {lab:<26} n={r['n_ids']:5d}  EER {pct(r['eer'])}  rank-1 {pct(r['rank1'])}")
    if "tstr" in m:
        print(f" Utility on {m['test_subjects']} UNSEEN real subjects (LDA embedding trained on ...)")
        for k, r in m["tstr"].items():
            print(f"   {k:<16} EER {pct(r['eer'])} rank-1 {pct(r['rank1'])} | cross-state EO->EC: "
                  f"EER {pct(r['eer_cross_state'])} rank-1 {pct(r['rank1_cross_state'])}")
    f = m["fidelity"]
    print(" Fidelity")
    print(f"   discriminator AUC real-vs-syn {f['discriminator_auc']:.3f} (0.5 = indistinguishable)")
    print(f"   Frechet dist syn {f['frechet_syn_vs_train']:.2f} | unseen-real-people reference {f['frechet_unseen_real_vs_train(ref)']:.2f}")
    print(f"   MSE means {f['mse_means']:.4f} | MSE correlations {f['mse_correlation']:.4f} | std ratio {f['std_ratio']:.3f}")
    print(f"   between-person variance share real {f['between_person_var_share_real']:.3f} | syn {f['between_person_var_share_syn']:.3f}")
    if "state_task" in m:
        s = m["state_task"]
        print(" Eyes-open vs eyes-closed XGBoost (tested on unseen real subjects)")
        print(f"   real {pct(s['xgb_acc_real->real'])} | synthetic {pct(s['xgb_acc_syn->real(TSTR)'])} | "
              f"combined {pct(s['xgb_acc_combined->real'])}")
        if "shap_top10_similarity" in s:
            print(f"   SHAP top-10 feature similarity {s['shap_top10_similarity'] * 100:.0f}%")
    p = m["privacy"]
    print(" Privacy")
    print(f"   median NN distance to a real person = {p['nn_dist_to_real_over_real_median']:.2f} x real-real median")
    print(f"   synthetic ids closer than the closest real pair: {pct(p['frac_closer_than_closest_real_pair'])}")
    print(f"   re-identified as a real training subject:       {pct(p['reid_rate_vs_real_train'])}")
