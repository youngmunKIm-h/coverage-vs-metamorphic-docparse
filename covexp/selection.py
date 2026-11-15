"""테스트 집합 선별 전략."""
import heapq
from collections import defaultdict

import numpy as np


def greedy_cov(feats, cand, n, M, rng):
    """커버리지 증가량이 가장 큰 입력을 차례로 추가 (lazy greedy).
    더 늘릴 커버리지가 없으면 나머지는 무작위로 채움."""
    covered = np.zeros(M, dtype=bool)
    heap = [(-len(feats[i]), rng.random(), i) for i in cand]
    heapq.heapify(heap)
    chosen = []
    while heap and len(chosen) < n:
        neg, tb, i = heapq.heappop(heap)
        g = int((~covered[feats[i]]).sum())
        if g == 0:
            rest = [i] + [x[2] for x in heap]
            rng.shuffle(rest)
            chosen += rest[: n - len(chosen)]
            break
        if heap and g < -heap[0][0]:
            heapq.heappush(heap, (-g, tb, i))
            continue
        chosen.append(i)
        covered[feats[i]] = True
    return chosen


def cov_low(feats, cand, n, rng):
    """개별 커버리지가 가장 작은 입력부터 (역방향 대조군)."""
    order = sorted(cand, key=lambda i: (len(feats[i]), rng.random()))
    return order[:n]


def random_sel(cand, n, rng):
    c = list(cand)
    rng.shuffle(c)
    return c[:n]


def ep_sel(partition, cand, n, rng):
    """동등 클래스별로 고르게 (라운드로빈)."""
    groups = defaultdict(list)
    for i in cand:
        groups[partition[i]].append(i)
    keys = list(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    chosen = []
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop())
    return chosen


def top_score(scores, cand, n, rng):
    """점수(출력 불일치)가 큰 입력부터. 동점은 무작위."""
    return sorted(cand, key=lambda i: (-scores[i], rng.random()))[:n]
