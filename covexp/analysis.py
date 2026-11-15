"""통계: A12 효과 크기, 맨-휘트니 U, 홈 보정, 스피어만."""
import numpy as np
from scipy import stats


def a12(x, y):
    """x에서 하나, y에서 하나 뽑았을 때 x가 더 클 확률 (동점은 0.5)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def mwu_p(x, y):
    if np.unique(np.concatenate([np.asarray(x, float), np.asarray(y, float)])).size == 1:
        return 1.0   # 모든 값이 같으면 검정 불가 → 차이 없음
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def holm(pvals):
    """다중 비교 보정. NaN(검정 불가)은 그대로 NaN."""
    p = np.asarray(pvals, float)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m, running = len(ok), 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def spearman(x, y):
    r = stats.spearmanr(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r.correlation), float(r.pvalue)


def a12_label(v):
    d = abs(v - 0.5)
    return "무시" if d < 0.06 else "작음" if d < 0.14 else "중간" if d < 0.21 else "큼"
