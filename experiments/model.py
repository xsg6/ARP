"""Lightweight trainable pooling and scoring on cached, frozen model states."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class Ranker:
    def __init__(self, mode, seed, rank_weight=1.0):
        self.mode = mode
        self.seed = seed
        self.rank_weight = rank_weight

    def fit(self, x, h, y, bugs):
        # x is the edit representation; h contains token-position states of both LMs.
        self.xmean, self.xstd = x.mean(0), x.std(0).clip(1e-4)
        self.hmean = h.mean(axis=(0, 1))
        self.hstd = h.std(axis=(0, 1)).clip(1e-4)
        x, h = self.normalize(x, h)
        rng = np.random.default_rng(self.seed)
        groups = [np.flatnonzero(bugs == b) for b in np.unique(bugs)]
        # Bug-level bagging supplies independent ensemble heads.
        groups = [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        ids = np.concatenate(groups)
        if len(np.unique(y[ids])) < 2:
            groups = [np.flatnonzero(bugs == b) for b in np.unique(bugs)]
            ids = np.concatenate(groups)
        pos, neg = [], []
        for g in groups:
            a, b = g[y[g] == 1], g[y[g] == 0]
            if len(a) and len(b):
                # Bound the number of pairs per bug so large pools do not dominate.
                pos.extend(rng.choice(a, 32))
                neg.extend(rng.choice(b, 32))
        pos, neg = np.array(pos, int), np.array(neg, int)
        counts = np.bincount(y[ids], minlength=2)
        if min(counts) == 0:
            raise ValueError("training needs both correctness classes")
        weights = 1 / (2 * counts[y[ids]])
        d, k = x.shape[1], h.shape[2]

        def objective(theta):
            w, v, a, bias = self.unpack(theta, d, k)
            pooled, att = self.pool(h, a)
            logits = x @ w + pooled @ v + bias
            z = logits[ids]
            loss = weights @ (np.logaddexp(0, z) - y[ids] * z)
            grad = np.bincount(ids, weights * (expit(z) - y[ids]), minlength=len(y))
            if len(pos) and self.rank_weight:
                delta = logits[pos] - logits[neg]
                loss += self.rank_weight * np.logaddexp(0, -delta).mean()
                pair_grad = -self.rank_weight * expit(-delta) / len(pos)
                np.add.at(grad, pos, pair_grad)
                np.add.at(grad, neg, -pair_grad)
            da = np.zeros(k)
            if self.mode in {"fusion", "probe"}:
                token_scores = h @ v
                centered = token_scores - (att * token_scores).sum(1, keepdims=True)
                da = np.einsum("nt,ntd->d", grad[:, None] * att * centered, h) / np.sqrt(k)
            gradient = np.r_[x.T @ grad, pooled.T @ grad, da, grad.sum()]
            reg = 0.015
            gradient[:-1] += reg * theta[:-1]
            return loss + reg / 2 * (theta[:-1] @ theta[:-1]), gradient

        result = minimize(objective, rng.normal(0, 0.001, d + 2*k + 1),
                          jac=True, method="L-BFGS-B", options={"maxiter": 200})
        if not result.success:
            raise RuntimeError(result.message)
        self.theta = result.x
        return self

    @staticmethod
    def unpack(t, d, k):
        return t[:d], t[d:d+k], t[d+k:d+2*k], t[-1]

    def normalize(self, x, h):
        x = ((x - self.xmean) / self.xstd).clip(-10, 10)
        h = ((h - self.hmean) / self.hstd).clip(-10, 10)
        if self.mode == "probe":
            x = np.zeros((len(x), 0))
        return x, h

    def pool(self, h, a):
        if self.mode == "diff":
            return np.zeros((len(h), h.shape[-1])), None
        if self.mode == "last":
            return h[:, -1], None
        if self.mode == "mean":
            return h.mean(1), None
        z = (h @ a) / np.sqrt(len(a))
        att = np.exp(z - z.max(1, keepdims=True))
        att /= att.sum(1, keepdims=True)
        return np.einsum("nt,ntd->nd", att, h), att

    def logits(self, x, h):
        x, h = self.normalize(x, h)
        w, v, a, b = self.unpack(self.theta, x.shape[1], h.shape[-1])
        return x @ w + self.pool(h, a)[0] @ v + b
