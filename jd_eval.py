"""Ranking evaluation shared by steps 3 and 4. Written by step3_baselines.ipynb: edit it there.

Every metric is computed per slate (one user's test query) and then averaged over slates.
Candidates are ranked by score with ties broken at random, never by list position.
Gain is 2**label - 1 (click 1, cart 3, purchase 7).
NDCG and HR are NaN for a slate with no positive; purchase recall is NaN for a slate with no purchase.
"""
import numpy as np
import pandas as pd

METRICS = ["ndcg@5", "ndcg@10", "hr@10", "purchase_recall@10", "ndcg@10_cold", "ndcg@10_warm"]


def per_slate(user, label, score, k=(5, 10), seed=0):
    """One row per slate: NDCG@k for each k, HR@10 and purchase recall@10."""
    user = np.asarray(user, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    cols = ["user_id", "n"] + [f"ndcg@{x}" for x in k] + ["hr@10", "purchase_recall@10"]
    if len(user) == 0:
        return pd.DataFrame(columns=cols)
    tie = np.random.default_rng(seed).random(len(user))
    o = np.lexsort((tie, -score, user))                       # by slate, then score high to low
    u, y = user[o], label[o]
    start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
    n = np.diff(np.r_[start, len(u)])
    rank = np.arange(len(u)) - np.repeat(start, n)            # 0 = top of the slate
    disc = 1.0 / np.log2(rank + 2.0)
    gain = 2.0 ** y - 1.0
    ideal = 2.0 ** y[np.lexsort((-y, u))] - 1.0               # same slates, labels sorted high to low
    total = lambda x: np.add.reduceat(x, start)
    has_pos = total((y > 0).astype(float)) > 0
    out = pd.DataFrame({"user_id": u[start], "n": n})
    for kk in k:
        top = rank < kk
        dcg, idcg = total(gain * disc * top), total(ideal * disc * top)
        out[f"ndcg@{kk}"] = np.where(has_pos, dcg / np.where(idcg > 0, idcg, 1.0), np.nan)
    top10 = rank < 10
    out["hr@10"] = np.where(has_pos, (total(((y > 0) & top10).astype(float)) > 0).astype(float), np.nan)
    n_pur = total((y == 3).astype(float))
    out["purchase_recall@10"] = np.where(n_pur > 0, total(((y == 3) & top10).astype(float))
                                         / np.where(n_pur > 0, n_pur, 1.0), np.nan)
    return out


def evaluate(user, label, score, cold=None, seed=0):
    """per_slate, plus NDCG@10 inside the cold and the warm part of each slate when `cold` is given."""
    df = per_slate(user, label, score, seed=seed)
    if cold is not None:
        user, label, score = np.asarray(user), np.asarray(label), np.asarray(score)
        cold = np.asarray(cold, dtype=bool)
        for name, m in (("cold", cold), ("warm", ~cold)):
            s = per_slate(user[m], label[m], score[m], k=(10,), seed=seed)
            df = df.merge(s[["user_id", "ndcg@10"]].rename(columns={"ndcg@10": f"ndcg@10_{name}"}),
                          on="user_id", how="left")
    return df


def _ci(x, n_boot, rng, chunk=100):
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        means[s:s + b] = x[rng.integers(0, len(x), (b, len(x)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def summarize(df, metrics=METRICS, n_boot=1000, seed=0):
    """Mean of each metric over slates where it is defined, with a 95% bootstrap interval over slates."""
    rng, out = np.random.default_rng(seed), {}
    for m in metrics:
        if m not in df:
            continue
        x = df[m].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            out.update({m: np.nan, f"{m}_lo": np.nan, f"{m}_hi": np.nan, f"{m}_slates": 0})
            continue
        lo, hi = _ci(x, n_boot, rng)
        out.update({m: x.mean(), f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_slates": len(x)})
    return out


def paired(a, b, metric="ndcg@10", n_boot=1000, seed=0):
    """Mean of (a - b) over the slates both scored, with a paired bootstrap 95% interval."""
    m = a[["user_id", metric]].merge(b[["user_id", metric]], on="user_id", suffixes=("_a", "_b")).dropna()
    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    if len(d) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "slates": 0}
    lo, hi = _ci(d, n_boot, np.random.default_rng(seed))
    return {"diff": d.mean(), "lo": lo, "hi": hi, "slates": len(d)}
