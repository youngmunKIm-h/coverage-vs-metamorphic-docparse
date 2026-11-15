"""NC / KMNC / NBC. 입력마다 '덮은 특성 인덱스' 배열로 표현 → 집합 커버리지 = 합집합 크기 / 전체 특성 수."""
import numpy as np


def nc_features(acts, layer_slices, t):
    """DeepXplore 방식: 입력·레이어별 min-max 스케일 후 t 초과 뉴런."""
    scaled = np.empty_like(acts, dtype=np.float32)
    for lo, hi in layer_slices:
        a = acts[:, lo:hi].astype(np.float32)
        mn, mx = a.min(1, keepdims=True), a.max(1, keepdims=True)
        scaled[:, lo:hi] = (a - mn) / (mx - mn + 1e-12)
    feats = [np.flatnonzero(row > t).astype(np.int64) for row in scaled]
    return feats, acts.shape[1]


def kmnc_features(acts, lo, hi, k):
    """프로파일 범위 [lo, hi]를 k구간으로 나눠 (뉴런, 구간) 쌍을 특성으로."""
    N = acts.shape[1]
    width = hi - lo
    valid = width > 0
    a = acts.astype(np.float32)
    sec = np.floor((a - lo) / np.where(valid, width, 1.0) * k).astype(np.int64)
    sec = np.clip(sec, 0, k - 1)
    inr = (a >= lo) & (a <= hi) & valid
    feats = []
    for i in range(len(a)):
        n = np.flatnonzero(inr[i])
        feats.append(n * k + sec[i, n])
    return feats, N * k


def nbc_features(acts, lo, hi):
    """프로파일 범위 밖 극단 활성: 상한 초과 N개 + 하한 미만 N개."""
    N = acts.shape[1]
    a = acts.astype(np.float32)
    feats = [np.concatenate([np.flatnonzero(r > hi), N + np.flatnonzero(r < lo)]).astype(np.int64) for r in a]
    return feats, 2 * N


def set_coverage(idx, feats, M):
    covered = np.zeros(M, dtype=bool)
    for i in idx:
        covered[feats[i]] = True
    return covered.sum() / M


def build_all(acts, layer_slices, lo, hi, cfg):
    """{지표 이름: (특성 리스트, 전체 특성 수)}"""
    out = {}
    for t in cfg.nc_thresholds:
        out[f"NC@{t}"] = nc_features(acts, layer_slices, t)
    for k in cfg.kmnc_ks:
        out[f"KMNC@{k}"] = kmnc_features(acts, lo, hi, k)
    out["NBC"] = nbc_features(acts, lo, hi)
    return out
