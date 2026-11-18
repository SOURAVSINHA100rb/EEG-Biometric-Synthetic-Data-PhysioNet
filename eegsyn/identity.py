"""
Hierarchical identity model: how we get 2,500 NEW people from 109 real ones.

A conditional generator trained with "subject = class label" (as the original
project did with emotion labels) can only reproduce the people it has seen.
To create new, distinct identities we split every EEG window into

    x  =  mu(person, state)  +  r          (r = within-person variability)

1. mu: each real person is summarised by a prototype vector
   [mean EO features, mean EC features] (2 x 320 dims). PCA (whitened) of the
   prototypes gives a low-dimensional *identity code* c ~ N(0, I).
   New people are created by sampling new codes c, decoding them back to
   prototypes and adding a fixed person-specific residual drawn from the part
   of the prototype variance PCA does not explain.
   A privacy filter rejects codes that land too close to a real person.

2. r: one of the five generators (VAE / GAN / Diffusion / LSTM / Copula)
   learns p(r | c, state) - how a person's windows scatter around their own
   mean, which depends on who they are and on eyes-open vs eyes-closed.
"""
import numpy as np
from sklearn.decomposition import PCA


class IdentityModel:
    def __init__(self, n_components=32, seed=0):
        self.k = n_components
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ fit
    def fit(self, X, subjects, states, state_names):
        """X: standardised features (N, D); subjects: (N,), states: (N,) str."""
        self.state_names = list(state_names)
        self.D = X.shape[1]
        self.subject_ids = np.unique(subjects)
        protos = []
        for s in self.subject_ids:
            m = subjects == s
            protos.append(np.concatenate([X[m & (states == st)].mean(0) for st in self.state_names]))
        self.prototypes = np.stack(protos)                         # (S, 2D)
        k = min(self.k, len(self.subject_ids) - 1)
        self.pca = PCA(n_components=k, whiten=True, random_state=0).fit(self.prototypes)
        self.k = k
        self.codes = self.pca.transform(self.prototypes)            # (S, k)  ~N(0, I)
        recon = self.pca.inverse_transform(self.codes)
        self.proto_resid_std = (self.prototypes - recon).std(0)     # unexplained identity detail
        self.code_of = dict(zip(self.subject_ids, self.codes))
        self.proto_of = dict(zip(self.subject_ids, self.prototypes))
        # real-to-real nearest-neighbour distances in code space (privacy reference)
        d = np.linalg.norm(self.codes[:, None] - self.codes[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        self.real_nn = d.min(1)
        return self

    def explained_variance(self):
        return float(self.pca.explained_variance_ratio_.sum())

    def mean_for(self, prototype, state):
        i = self.state_names.index(state)
        return prototype[i * self.D:(i + 1) * self.D]

    def residuals(self, X, subjects, states):
        """Within-person residuals r = x - mu(subject, state) and conditioning codes."""
        R = np.empty_like(X)
        C = np.empty((len(X), self.k), dtype=np.float32)
        for s in self.subject_ids:
            for st in self.state_names:
                m = (subjects == s) & (states == st)
                R[m] = X[m] - self.mean_for(self.proto_of[s], st)
                C[m] = self.code_of[s]
        return R, C

    # --------------------------------------------------------------- sample
    def sample_identities(self, n, temperature=1.0, min_dist_quantile=0.05, max_tries=200):
        """Sample n new identity codes + prototypes.

        A candidate is rejected when it is closer to ANY real training person
        than the `min_dist_quantile` of real-vs-real nearest-neighbour
        distances (i.e. it would look like a near-copy of a real subject).
        """
        tau = np.quantile(self.real_nn, min_dist_quantile)
        codes = []
        tries = 0
        while len(codes) < n and tries < max_tries:
            cand = self.rng.normal(0, temperature, (n * 2, self.k))
            d = np.linalg.norm(cand[:, None] - self.codes[None], axis=-1).min(1)
            codes.extend(cand[d > tau][: n - len(codes)])
            tries += 1
        codes = np.asarray(codes[:n], dtype=np.float32)
        protos = self.pca.inverse_transform(codes)
        protos = protos + self.rng.normal(0, 1, protos.shape) * self.proto_resid_std
        self.privacy_tau = float(tau)
        return codes, protos.astype(np.float32)
