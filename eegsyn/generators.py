"""
The five synthetic-data generators of the original project, rewritten in
PyTorch and made *identity-conditional*.

Every generator models p(r | c, state):
    r     within-person residual of one EEG window (320 features)
    c     identity code of the person (see identity.py)
    state 0 = eyes open, 1 = eyes closed
Interface:  gen.fit(R, C, S)   ->   gen.sample(C, S) -> R_synthetic

Differences from the BED version (and why):
  * VAE      : Gaussian (MSE) reconstruction instead of binary cross-entropy,
               because log band powers are real-valued, not in [0, 1].
  * GAN      : WGAN-GP critic - far more stable than the vanilla GAN on 320-dim
               tabular data.
  * Diffusion: a real DDPM (forward noising + learned reverse process), the
               original was a single-step denoising auto-encoder.
  * LSTM     : a variational LSTM (sequence VAE) that walks over the 64
               electrodes, each step emitting that electrode's 5 band powers.
  * Copula   : Gaussian copula on [identity code, residual] with an exact
               conditional sampler, empirical marginals (no extra package).
During training, identity codes get small Gaussian jitter so the networks
learn a smooth function of c and generalise to the never-seen codes of the
2,500 synthetic people.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

torch.set_num_threads(max(1, torch.get_num_threads()))


def _mlp(i, h, o, n=3, act=nn.SiLU, norm_layer=True):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if norm_layer else []) + [act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class BaseGenerator:
    name = "base"

    def __init__(self, epochs=200, batch_size=256, lr=1e-3, code_jitter=0.3, seed=0, verbose=True):
        self.epochs, self.bs, self.lr = epochs, batch_size, lr
        self.jitter, self.seed, self.verbose = code_jitter, seed, verbose
        torch.manual_seed(seed)
        np.random.seed(seed)

    # residuals are rescaled to unit variance per feature for the networks
    def _prep(self, R, C, S):
        self.r_mu, self.r_sd = R.mean(0), R.std(0) + 1e-6
        self.k = C.shape[1]
        Rn = (R - self.r_mu) / self.r_sd
        return (torch.tensor(Rn, dtype=torch.float32), torch.tensor(C, dtype=torch.float32),
                torch.tensor(S, dtype=torch.long))

    def _cond(self, C, S, train):
        if train and self.jitter > 0:
            C = C + self.jitter * torch.randn_like(C)
        return torch.cat([C, F.one_hot(S, 2).float()], 1)

    def _unprep(self, Rn):
        return Rn * self.r_sd + self.r_mu

    def _loop(self, n, step_fn):
        t0 = time.time()
        for ep in range(1, self.epochs + 1):
            perm = torch.randperm(n)
            logs = []
            for i in range(0, n, self.bs):
                logs.append(step_fn(perm[i:i + self.bs]))
            if self.verbose and (ep % max(1, self.epochs // 10) == 0 or ep == 1):
                msg = " ".join(f"{k}={np.mean([l[k] for l in logs]):.4f}" for k in logs[0])
                print(f"  [{self.name}] epoch {ep:4d}/{self.epochs}  {msg}  ({time.time() - t0:.0f}s)", flush=True)


# ============================================================== 0. CVAE
class CVAE(BaseGenerator):
    name = "VAE"

    def __init__(self, latent_dim=32, hidden=512, beta=1.0, **kw):
        super().__init__(**kw)
        self.zd, self.h, self.beta = latent_dim, hidden, beta

    def fit(self, R, C, S):
        Rt, Ct, St = self._prep(R, C, S)
        D, cd = Rt.shape[1], self.k + 2
        self.enc = _mlp(D + cd, self.h, 2 * self.zd)
        self.dec = _mlp(self.zd + cd, self.h, D)
        opt = torch.optim.Adam([*self.enc.parameters(), *self.dec.parameters()], lr=self.lr)

        def step(idx):
            x, c = Rt[idx], self._cond(Ct[idx], St[idx], True)
            mu, logv = self.enc(torch.cat([x, c], 1)).chunk(2, 1)
            z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
            xh = self.dec(torch.cat([z, c], 1))
            rec = F.mse_loss(xh, x, reduction="none").sum(1).mean()
            kl = (-0.5 * (1 + logv - mu ** 2 - logv.exp()).sum(1)).mean()
            loss = rec + self.beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            return {"rec": rec.item() / D, "kl": kl.item()}

        self._loop(len(Rt), step)
        # VAE decoders output the conditional *mean*; estimate the per-feature
        # reconstruction noise so sampled windows are not over-smoothed
        with torch.no_grad():
            c = self._cond(Ct, St, False)
            mu, _ = self.enc(torch.cat([Rt, c], 1)).chunk(2, 1)
            self.noise_sd = (self.dec(torch.cat([mu, c], 1)) - Rt).std(0)
        return self

    @torch.no_grad()
    def sample(self, C, S):
        c = self._cond(torch.tensor(C, dtype=torch.float32), torch.tensor(S, dtype=torch.long), False)
        z = torch.randn(len(c), self.zd)
        out = self.dec(torch.cat([z, c], 1)) + torch.randn(len(c), len(self.r_sd)) * self.noise_sd
        return self._unprep(out.numpy())


# ============================================================== 1. WGAN-GP
class CGAN(BaseGenerator):
    name = "GAN"

    def __init__(self, noise_dim=64, hidden=512, n_critic=5, gp=10.0, **kw):
        kw.setdefault("lr", 2e-4)
        super().__init__(**kw)
        self.nz, self.h, self.n_critic, self.gp = noise_dim, hidden, n_critic, gp

    def fit(self, R, C, S):
        Rt, Ct, St = self._prep(R, C, S)
        D, cd = Rt.shape[1], self.k + 2
        self.G = _mlp(self.nz + cd, self.h, D, n=4)
        self.Dc = _mlp(D + cd, self.h, 1, n=4, act=lambda: nn.LeakyReLU(0.2), norm_layer=False)
        oG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        oD = torch.optim.Adam(self.Dc.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self._it = 0

        def step(idx):
            x, c = Rt[idx], self._cond(Ct[idx], St[idx], True)
            b = len(x)
            fake = self.G(torch.cat([torch.randn(b, self.nz), c], 1))
            # critic
            d_real = self.Dc(torch.cat([x, c], 1)).mean()
            d_fake = self.Dc(torch.cat([fake.detach(), c], 1)).mean()
            eps = torch.rand(b, 1)
            xi = (eps * x + (1 - eps) * fake.detach()).requires_grad_(True)
            g = torch.autograd.grad(self.Dc(torch.cat([xi, c], 1)).sum(), xi, create_graph=True)[0]
            gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
            lossD = d_fake - d_real + self.gp * gp
            oD.zero_grad(); lossD.backward(); oD.step()
            self._it += 1
            out = {"W": (d_real - d_fake).item()}
            if self._it % self.n_critic == 0:
                fake = self.G(torch.cat([torch.randn(b, self.nz), c], 1))
                lossG = -self.Dc(torch.cat([fake, c], 1)).mean()
                oG.zero_grad(); lossG.backward(); oG.step()
            return out

        self._loop(len(Rt), step)
        return self

    @torch.no_grad()
    def sample(self, C, S):
        c = self._cond(torch.tensor(C, dtype=torch.float32), torch.tensor(S, dtype=torch.long), False)
        return self._unprep(self.G(torch.cat([torch.randn(len(c), self.nz), c], 1)).numpy())


# ============================================================== 2. DDPM
class Diffusion(BaseGenerator):
    name = "Diffusion"

    def __init__(self, T=200, hidden=768, clip=4.0, ema_decay=0.999, **kw):
        super().__init__(**kw)
        self.T, self.h, self.clip, self.ema_decay = T, hidden, clip, ema_decay
        # cosine noise schedule (Nichol & Dhariwal 2021)
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.betas = (1 - ac[1:] / ac[:-1]).clamp(1e-5, 0.999)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def _temb(self, t, dim=64):
        half = dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], 1)

    def fit(self, R, C, S):
        Rt, Ct, St = self._prep(R, C, S)
        D, cd = Rt.shape[1], self.k + 2
        self.net = _mlp(D + cd + 64, self.h, D, n=4)
        self.ema = _mlp(D + cd + 64, self.h, D, n=4)
        self.ema.load_state_dict(self.net.state_dict())
        self.ema.requires_grad_(False)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs * math.ceil(len(Rt) / self.bs))

        def step(idx):
            x0, c = Rt[idx], self._cond(Ct[idx], St[idx], True)
            # classifier-free guidance training: drop the condition 10 % of the time
            drop = (torch.rand(len(c), 1) < 0.1).float()
            c = c * (1 - drop)
            t = torch.randint(0, self.T, (len(x0),))
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            loss = F.mse_loss(self.net(torch.cat([xt, c, self._temb(t)], 1)), eps)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():   # exponential moving average of weights, used for sampling
                for pe, pn in zip(self.ema.parameters(), self.net.parameters()):
                    pe.mul_(self.ema_decay).add_(pn, alpha=1 - self.ema_decay)
            return {"mse": loss.item()}

        self._loop(len(Rt), step)
        return self

    @torch.no_grad()
    def sample(self, C, S, guidance=1.0, chunk=8192):
        outs = []
        for i in range(0, len(C), chunk):
            c = self._cond(torch.tensor(C[i:i + chunk], dtype=torch.float32),
                           torch.tensor(S[i:i + chunk], dtype=torch.long), False)
            x = torch.randn(len(c), len(self.r_sd))
            for t in reversed(range(self.T)):
                tt = torch.full((len(c),), t, dtype=torch.long)
                te = self._temb(tt)
                e = self.ema(torch.cat([x, c, te], 1))
                if guidance != 1.0:
                    e0 = self.ema(torch.cat([x, torch.zeros_like(c), te], 1))
                    e = e0 + guidance * (e - e0)
                # predict x0, clip it to a plausible range, then take the DDPM posterior
                # mean q(x_{t-1} | x_t, x0). Without the clip the large final betas of the
                # cosine schedule amplify prediction error and sampling diverges.
                a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
                ab_prev = self.abar[t - 1] if t > 0 else torch.tensor(1.0)
                x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-self.clip, self.clip)
                x = (ab_prev.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ab_prev) / (1 - ab)) * x
                if t > 0:
                    var = b * (1 - ab_prev) / (1 - ab)
                    x = x + var.sqrt() * torch.randn_like(x)
            outs.append(x.numpy())
        return self._unprep(np.concatenate(outs))


# ============================================================== 3. Variational LSTM
class VLSTM(BaseGenerator):
    name = "LSTM"

    def __init__(self, n_channels=64, n_bands=5, latent_dim=32, hidden=128, beta=1.0, **kw):
        kw.setdefault("lr", 2e-3)
        super().__init__(**kw)
        self.nc, self.nb, self.zd, self.h, self.beta = n_channels, n_bands, latent_dim, hidden, beta

    def _build(self, cd):
        self.enc = nn.LSTM(self.nb + cd, self.h, batch_first=True, bidirectional=True)
        self.enc_out = nn.Linear(2 * self.h, 2 * self.zd)
        self.ch_emb = nn.Embedding(self.nc, 16)
        self.dec = nn.LSTM(self.zd + cd + 16, self.h, num_layers=2, batch_first=True, dropout=0.1)
        self.dec_out = nn.Linear(self.h, self.nb)
        self.params = [*self.enc.parameters(), *self.enc_out.parameters(), *self.ch_emb.parameters(),
                       *self.dec.parameters(), *self.dec_out.parameters()]

    def _decode(self, z, c):
        b = len(z)
        ch = self.ch_emb(torch.arange(self.nc))[None].expand(b, -1, -1)
        inp = torch.cat([torch.cat([z, c], 1)[:, None].expand(-1, self.nc, -1), ch], 2)
        return self.dec_out(self.dec(inp)[0]).reshape(b, -1)

    def fit(self, R, C, S):
        Rt, Ct, St = self._prep(R, C, S)
        cd = self.k + 2
        self._build(cd)
        opt = torch.optim.Adam(self.params, lr=self.lr)

        def step(idx):
            x, c = Rt[idx], self._cond(Ct[idx], St[idx], True)
            seq = torch.cat([x.reshape(len(x), self.nc, self.nb), c[:, None].expand(-1, self.nc, -1)], 2)
            hseq = self.enc(seq)[0]
            mu, logv = self.enc_out(hseq.mean(1)).chunk(2, 1)
            z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
            xh = self._decode(z, c)
            rec = F.mse_loss(xh, x, reduction="none").sum(1).mean()
            kl = (-0.5 * (1 + logv - mu ** 2 - logv.exp()).sum(1)).mean()
            loss = rec + self.beta * kl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.params, 5.0); opt.step()
            return {"rec": rec.item() / x.shape[1], "kl": kl.item()}

        self._loop(len(Rt), step)
        # residual decoder noise (same idea as the CVAE)
        with torch.no_grad():
            c = self._cond(Ct, St, False)
            seq = torch.cat([Rt.reshape(len(Rt), self.nc, self.nb), c[:, None].expand(-1, self.nc, -1)], 2)
            mu, _ = self.enc_out(self.enc(seq)[0].mean(1)).chunk(2, 1)
            self.noise_sd = (self._decode(mu, c) - Rt).std(0)
        return self

    @torch.no_grad()
    def sample(self, C, S, chunk=1024):
        outs = []
        for i in range(0, len(C), chunk):
            c = self._cond(torch.tensor(C[i:i + chunk], dtype=torch.float32),
                           torch.tensor(S[i:i + chunk], dtype=torch.long), False)
            z = torch.randn(len(c), self.zd)
            outs.append((self._decode(z, c) + torch.randn(len(c), self.nc * self.nb) * self.noise_sd).numpy())
        return self._unprep(np.concatenate(outs))


# ============================================================== 4. Gaussian copula
class GaussianCopula(BaseGenerator):
    """Per state: Gaussian copula over [c, r].  Sampling r | c uses the exact
    Gaussian conditional in normal-score space, then empirical inverse CDFs."""
    name = "Copula"

    def __init__(self, n_quantiles=1000, **kw):
        super().__init__(**kw)
        self.nq = n_quantiles

    def fit(self, R, C, S):
        self.k = C.shape[1]
        self.models = {}
        qs = np.linspace(0, 1, self.nq)
        for s in (0, 1):
            Rs, Cs = R[S == s], C[S == s]
            n = len(Rs)
            ranks = Rs.argsort(0).argsort(0)
            Z = norm.ppf((ranks + 0.5) / n)                       # normal scores of residuals
            J = np.concatenate([Cs, Z], 1)                        # codes are already ~N(0,1)
            cov = LedoitWolf().fit(J).covariance_
            k = self.k
            Scc, Scr, Srr = cov[:k, :k], cov[:k, k:], cov[k:, k:]
            A = np.linalg.solve(Scc, Scr).T                       # E[z_r | c] = A c
            cond_cov = Srr - A @ Scr
            L = np.linalg.cholesky(cond_cov + 1e-6 * np.eye(len(cond_cov)))
            self.models[s] = dict(A=A, L=L, q=np.quantile(Rs, qs, axis=0))
            if self.verbose:
                print(f"  [Copula] state {s}: fitted {J.shape[1]}-dim Gaussian copula on {n} windows")
        return self

    def sample(self, C, S):
        out = np.empty((len(C), self.models[0]["q"].shape[1]), dtype=np.float32)
        qs = np.linspace(0, 1, self.nq)
        for s in (0, 1):
            m = S == s
            if not m.any():
                continue
            M = self.models[s]
            z = C[m] @ M["A"].T + np.random.randn(m.sum(), M["L"].shape[0]) @ M["L"].T
            u = norm.cdf(z)
            out[m] = np.stack([np.interp(u[:, j], qs, M["q"][:, j]) for j in range(u.shape[1])], 1)
        return out


GENERATORS = {"VAE": CVAE, "GAN": CGAN, "Diffusion": Diffusion, "LSTM": VLSTM, "Copula": GaussianCopula}
