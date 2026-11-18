"""
Evaluation of synthetic EEG *biometric* data.

The original project only checked "does an XGBoost trained on synthetic data
classify the 4 emotions?". For biometrics we need more:

1. Biometric quality of the synthetic people          (are they distinct + consistent?)
   closed-set rank-1 identification and verification EER among the 2,500
   synthetic identities, compared with the same numbers on real people.
2. Utility: Train-on-Synthetic, Test-on-Real (TSTR)   (is it useful for training?)
   an LDA embedding is learned from (a) real train subjects, (b) only the
   2,500 synthetic identities, (c) both, and verification EER / rank-1 is
   measured on REAL subjects that neither the generator nor the embedding saw.
3. Fidelity                                           (does it look like real EEG?)
   per-feature mean/std error, Frechet distance, real-vs-synthetic
   discriminator AUC (0.5 = indistinguishable), between-person variance share,
   + the original project's XGBoost / SHAP top-10 feature similarity analysis
   on the eyes-open vs eyes-closed task.
4. Privacy                                            (are they copies of real people?)
   distance of each synthetic identity to its nearest real person, and the
   share of synthetic people a real-subject matcher would accept as a real
   training subject.
"""
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------- embedding
class LDAEmbedder:
    """Shrinkage LDA: a classic, strong baseline for EEG-biometric templates."""

    def __init__(self, n_components=None):
        self.nc = n_components

    def fit(self, X, y):
        n = min(len(np.unique(y)) - 1, X.shape[1], self.nc or 10 ** 9)
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=n).fit(X, y)
        return self

    def __call__(self, X):
        E = self.lda.transform(X)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def eer_from_scores(gen, imp):
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    fpr, tpr, thr = roc_curve(y, np.r_[gen, imp])
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def first_half(subj, states, order):
    """Boolean mask: True for the temporally first half of every (person, state) recording.
    Splitting by time (not at random) avoids leakage between overlapping windows."""
    mask = np.zeros(len(subj), bool)
    for s in np.unique(subj):
        for st in np.unique(states):
            idx = np.where((subj == s) & (states == st))[0]
            if len(idx):
                idx = idx[np.argsort(order[idx], kind="stable")]
                mask[idx[: (len(idx) + 1) // 2]] = True
    return mask


def verify(embed, X, subj, states, order, protocol="temporal", rng=None, max_imp=2_000_000, chunk=4096):
    """Template-based verification + closed-set identification.

    protocol 'temporal'   : enrol on the first half of each EO and EC recording, probe with the second half
    protocol 'cross_state': enrol on eyes-open windows, probe with eyes-closed windows
    Returns dict(eer, rank1, genuine, impostor).
    """
    rng = rng or np.random.default_rng(0)
    E = embed(X)
    ids = np.unique(subj)
    enrol_mask = first_half(subj, states, order) if protocol == "temporal" else (states == "EO")
    templ, probes, probe_lab = [], [], []
    for j, s in enumerate(ids):
        idx = np.where(subj == s)[0]
        enr, prb = idx[enrol_mask[idx]], idx[~enrol_mask[idx]]
        t = E[enr].mean(0)
        templ.append(t / np.linalg.norm(t))
        probes.append(prb)
        probe_lab.append(np.full(len(prb), j))
    T = np.stack(templ)
    P, L = np.concatenate(probes), np.concatenate(probe_lab)
    gen, imp, correct = [], [], 0
    keep_imp = max_imp / max(1, len(P) * (len(ids) - 1))
    for i in range(0, len(P), chunk):
        sc = E[P[i:i + chunk]] @ T.T
        lab = L[i:i + chunk]
        rows = np.arange(len(lab))
        gen.append(sc[rows, lab])
        correct += (sc.argmax(1) == lab).sum()
        mask = np.ones_like(sc, dtype=bool)
        mask[rows, lab] = False
        im = sc[mask]
        if keep_imp < 1:
            im = im[rng.random(len(im)) < keep_imp]
        imp.append(im)
    gen, imp = np.concatenate(gen), np.concatenate(imp)
    eer, thr = eer_from_scores(gen, imp)
    return dict(eer=eer, rank1=float(correct / len(P)), threshold=thr, genuine=gen, impostor=imp,
                n_ids=len(ids))


# ----------------------------------------------------------------- fidelity
def frechet(A, B):
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def between_person_share(X, subj):
    """Average (over features) share of variance explained by person identity (ICC-like)."""
    tot = X.var(0)
    means = np.stack([X[subj == s].mean(0) for s in np.unique(subj)])
    return float(np.mean(means.var(0) / (tot + 1e-12)))


def discriminator_auc(Xr, Xs, rng, n=6000):
    a = Xr[rng.choice(len(Xr), min(n, len(Xr)), replace=False)]
    b = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    X = np.r_[a, b]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean())


def fidelity(Xr_train, Xr_test, Xs, subj_r, subj_s, rng):
    pca = PCA(50, random_state=0).fit(Xr_train)
    pr, pt, ps = pca.transform(Xr_train), pca.transform(Xr_test), pca.transform(Xs)
    corr_r = np.corrcoef(Xr_train, rowvar=False)
    corr_s = np.corrcoef(Xs, rowvar=False)
    return {
        "mean_abs_err": float(np.abs(Xr_train.mean(0) - Xs.mean(0)).mean()),
        "std_ratio": float(np.mean(Xs.std(0) / Xr_train.std(0))),
        "mse_means": float(((Xr_train.mean(0) - Xs.mean(0)) ** 2).mean()),
        "mse_correlation": float(((corr_r - corr_s) ** 2).mean()),
        "frechet_syn_vs_train": frechet(pr, ps),
        "frechet_unseen_real_vs_train(ref)": frechet(pr, pt),
        "discriminator_auc": discriminator_auc(Xr_train, Xs, rng),
        "between_person_var_share_real": between_person_share(Xr_train, subj_r),
        "between_person_var_share_syn": between_person_share(Xs, subj_s),
    }


# ------------------------------------------------------------------ privacy
def privacy(identity_model, syn_codes, embed_real, Xr, subj_r, Xs, subj_s, rng):
    """(i) code-space distance to nearest real person, (ii) matcher-based re-identification."""
    d = np.linalg.norm(syn_codes[:, None] - identity_model.codes[None], axis=-1).min(1)
    ref = np.median(identity_model.real_nn)

    def templates(X, subj):
        E = embed_real(X)
        ids = np.unique(subj)
        T = np.stack([E[subj == s].mean(0) for s in ids])
        return T / np.linalg.norm(T, axis=1, keepdims=True)

    # genuine reference: two half-templates of the same real person
    halves_a, halves_b = [], []
    E = embed_real(Xr)
    for s in np.unique(subj_r):
        idx = rng.permutation(np.where(subj_r == s)[0])
        a, b = E[idx[: len(idx) // 2]].mean(0), E[idx[len(idx) // 2:]].mean(0)
        halves_a.append(a / np.linalg.norm(a)); halves_b.append(b / np.linalg.norm(b))
    genuine = (np.stack(halves_a) * np.stack(halves_b)).sum(1)
    thr = np.quantile(genuine, 0.05)          # 95 % of real people would be accepted at this threshold
    Tr, Ts = templates(Xr, subj_r), templates(Xs, subj_s)
    max_sim = (Ts @ Tr.T).max(1)
    return {
        "nn_dist_to_real_over_real_median": float(np.median(d) / ref),
        "min_nn_dist_to_real_over_real_median": float(d.min() / ref),
        "frac_closer_than_closest_real_pair": float(np.mean(d < identity_model.real_nn.min())),
        "reid_rate_vs_real_train": float(np.mean(max_sim > thr)),
        "reid_threshold": float(thr),
    }


# --------------------------------------------- XGBoost / SHAP (original-style)
def state_task(Xr_train, yr_train, Xr_test, yr_test, Xs, ys, feat_names, rng, out_prefix, top_n=10, max_syn=20000):
    """Eyes-open vs eyes-closed: XGBoost accuracy (real / synthetic / combined -> real test)
    and SHAP top-10 feature similarity, exactly the analysis of the original repo."""
    from xgboost import XGBClassifier

    if len(Xs) > max_syn:
        sel = rng.choice(len(Xs), max_syn, replace=False)
        Xs, ys = Xs[sel], ys[sel]

    def fit(X, y):
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.5, eval_metric="logloss", n_jobs=-1).fit(X, y)

    m_real, m_syn = fit(Xr_train, yr_train), fit(Xs, ys)
    m_comb = fit(np.r_[Xr_train, Xs], np.r_[yr_train, ys])
    res = {
        "xgb_acc_real->real": float(accuracy_score(yr_test, m_real.predict(Xr_test))),
        "xgb_acc_syn->real(TSTR)": float(accuracy_score(yr_test, m_syn.predict(Xr_test))),
        "xgb_acc_combined->real": float(accuracy_score(yr_test, m_comb.predict(Xr_test))),
    }
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tops = {}
        for tag, model, X in (("Original", m_real, Xr_train), ("Synthetic", m_syn, Xs)):
            Xe = X[rng.choice(len(X), min(2000, len(X)), replace=False)]
            sv = shap.TreeExplainer(model).shap_values(Xe)
            if isinstance(sv, list):
                sv = sv[0]
            imp = np.abs(sv).mean(0)
            tops[tag] = [feat_names[i] for i in np.argsort(imp)[::-1][:top_n]]
            plt.figure()
            shap.summary_plot(sv, Xe, feature_names=feat_names, show=False, max_display=15)
            plt.title(f"SHAP Summary Plot for {tag} Data (EO vs EC)")
            plt.savefig(f"{out_prefix}_{tag}_SHAP_Summary_Plot.jpg", dpi=200, bbox_inches="tight")
            plt.savefig(f"{out_prefix}_{tag}_SHAP_Summary_Plot.pdf", bbox_inches="tight")
            plt.close()
        common = set(tops["Original"]) & set(tops["Synthetic"])
        res["shap_top10_similarity"] = len(common) / top_n
        res["shap_top10_original"] = tops["Original"]
        res["shap_top10_synthetic"] = tops["Synthetic"]
    except Exception as e:  # SHAP is optional
        print(f"  (SHAP analysis skipped: {e})")
    return res
