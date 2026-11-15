"""실험 파이프라인 (메인 환경). 단계별 실행:
  python run.py build                    # 데이터 표본 추출 + 변형 입력 풀 생성
  python run.py surrogate                # 대리 모델 활성 프로파일링 + 풀 추론
  (대상 환경에서) python run_target.py docling      # → work/target_pred_docling.json
  python run.py judge   --tgt docling    # TEDS·CER 계산 + 결함 판정
  python run.py rq1     --tgt docling    # RQ1: 커버리지-결함 상관
  python run.py select  --tgt docling    # RQ2/RQ3: 전략별 선별 반복 실험
  python run.py analyze --tgt docling    # RQ2/RQ3 통계 분석
"""
import argparse
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from covexp.config import Config
from covexp.data import reservoir_sample, gt_html, table_features
from covexp.transforms import TRANSFORMS, apply
from covexp.teds import teds, table_text, cer, wer
from covexp.coverage import build_all, set_coverage
from covexp import selection as S
from covexp.analysis import a12, a12_label, mwu_p, holm, spearman

cfg = Config()
W = cfg.work_dir
TGT = "docling"          # --tgt 로 덮어씀


def F(name):
    """대상 모델별 산출물 경로: judged_docling.csv 등."""
    stem, ext = name.rsplit(".", 1)
    return W / f"{stem}_{TGT}.{ext}"


def stage_build(args):
    jsonl = cfg.pubtabnet_dir / "PubTabNet_2.0.0.jsonl"
    (W / "pool").mkdir(parents=True, exist_ok=True)
    prof = reservoir_sample(jsonl, "train", cfg.n_profile, cfg.seed, cfg.pubtabnet_dir / "train")
    (W / "profile.txt").write_text(
        "\n".join(str(cfg.pubtabnet_dir / "train" / a["filename"]) for a in prof), encoding="utf-8")
    docs = reservoir_sample(jsonl, "val", cfg.n_docs, cfg.seed + 1, cfg.pubtabnet_dir / "val")
    gt, rows = {}, []
    for d, ann in enumerate(tqdm(docs, desc="pool")):
        gt[d] = gt_html(ann)
        f = table_features(ann)
        img = Image.open(cfg.pubtabnet_dir / "val" / ann["filename"]).convert("RGB")
        variants = [("none", "none", 0)] + [(t, TRANSFORMS[t][0], s) for t in TRANSFORMS for s in cfg.severities]
        for t, cat, s in variants:
            pid = f"d{d:04d}_{t}_{s}"
            out = img if t == "none" else apply(t, img, s, f"{cfg.seed}-{d}-{t}-{s}")
            out.save(W / "pool" / f"{pid}.png")
            rows.append({"id": pid, "doc": d, "src": ann["filename"], "transform": t, "category": cat,
                         "severity": s, **f,
                         "partition": f"{f['rows_b']}|{f['cols_b']}|{f['span']}|{cat}|{s}"})
    (W / "gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(W / "pool.csv", index=False)
    print(f"원본 {len(docs)}건, 풀 {len(rows)}건 생성")


def stage_surrogate(args):
    from covexp.surrogate import Surrogate, dets_to_html
    sur = Surrogate(cfg.model_name, cfg.det_threshold, cfg.max_side)
    print(f"device={sur.device}, 후킹 레이어 {len(sur.layer_names)}개")
    prof = (W / "profile.txt").read_text(encoding="utf-8").split("\n")
    A = np.stack([sur.run(Image.open(p))[0] for p in tqdm(prof, desc="profile")])
    np.save(W / "prof_lo.npy", A.min(0))
    np.save(W / "prof_hi.npy", A.max(0))
    pool = pd.read_csv(W / "pool.csv", keep_default_na=False)
    acts, preds = [], {}
    for pid in tqdm(pool["id"], desc="pool"):
        a, dets = sur.run(Image.open(W / "pool" / f"{pid}.png"))
        acts.append(a)
        preds[pid] = dets_to_html(dets)
    np.save(W / "acts_pool.npy", np.stack(acts))
    (W / "layer_slices.json").write_text(json.dumps(sur.layer_slices), encoding="utf-8")
    (W / "surrogate_pred.json").write_text(json.dumps(preds), encoding="utf-8")
    print(f"뉴런 수 N = {acts[0].shape[0]}, 표 미검출 {sum(v == '' for v in preds.values())}건")


def _is_defect(score, base=None):
    """결함 = 정답 HTML 대비 TEDS가 기준(teds_abs) 미만. 원본 출력과의 비교는 쓰지 않음."""
    return score < cfg.teds_abs


def stage_judge(args):
    pool = pd.read_csv(W / "pool.csv", keep_default_na=False)
    gt = json.loads((W / "gt.json").read_text(encoding="utf-8"))
    sp = json.loads((W / "surrogate_pred.json").read_text(encoding="utf-8"))
    pred_path = W / f"target_pred_{TGT}.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} 없음. 대상 환경에서 run_target.py {TGT} 를 먼저 실행하세요.")
    tg = json.loads(pred_path.read_text(encoding="utf-8"))
    missing = [i for i in pool["id"] if i not in tg]
    if missing:
        raise RuntimeError(f"대상 모델 결과 없는 입력 {len(missing)}건 (예: {missing[:3]}). run_target.py 를 끝까지 실행하세요.")
    rows = []
    for r in tqdm(pool.itertuples(), total=len(pool), desc="TEDS"):
        g = gt[str(r.doc)]
        rows.append({"id": r.id,
                     "sur_struct": teds(sp[r.id], g, True),
                     "tg_struct": teds(tg[r.id], g, True),
                     "tg_full": teds(tg[r.id], g, False),
                     "sur_empty": sp[r.id] == "", "tg_empty": tg[r.id] == ""})
    df = pool.merge(pd.DataFrame(rows), on="id")
    base = df[df["transform"] == "none"].set_index("doc")
    for col in ("sur_struct", "tg_struct", "tg_full"):
        df[col + "_base"] = df["doc"].map(base[col])
    # 분석 대상 문서: 원본 입력이 정답 기준을 통과한 문서만 (표 검출 + 구조·전체 TEDS ≥ teds_abs, 모두 정답 대비)
    tau = cfg.teds_abs
    b_empty = base["tg_empty"].astype(bool)
    b_struct_fail = ~b_empty & (base["tg_struct"] < tau)
    b_full_fail = ~b_empty & ~b_struct_fail & (base["tg_full"] < tau)
    bad_docs = sorted(base.index[b_empty | b_struct_fail | b_full_fail])
    df["base_ok"] = ~df["doc"].isin(bad_docs)
    excl = {"표 미검출": int(b_empty.sum()), "구조 TEDS 미달": int(b_struct_fail.sum()), "전체 TEDS 미달": int(b_full_fail.sum())}

    # 메타모픽 불일치: 같은 문서의 원본 입력에 대한 대상 모델 출력을 기준으로 (정답 라벨 불필요)
    base_id = df["doc"].map(base["id"])
    txt = {i: table_text(tg[i]) for i in df["id"]}
    df["mr_cer"] = [cer(txt[i], txt[b]) for i, b in zip(df["id"], base_id)]
    df["mr_wer"] = [wer(txt[i], txt[b]) for i, b in zip(df["id"], base_id)]
    df["mr_teds"] = [(0.0 if tg[i] == "" else 1.0) if tg[b] == "" else 1.0 - teds(tg[i], tg[b], True)
                     for i, b in zip(df["id"], base_id)]
    # 정답 대비 CER/WER (보고용)
    df["tg_cer_gt"] = [cer(txt[i], table_text(gt[str(d)])) for i, d in zip(df["id"], df["doc"])]
    df["tg_wer_gt"] = [wer(txt[i], table_text(gt[str(d)])) for i, d in zip(df["id"], df["doc"])]

    def sur_type(r):
        if r.sur_empty: return "missing"
        return "structure" if _is_defect(r.sur_struct, r.sur_struct_base) else ""

    def tg_type(r):
        if r.tg_empty: return "missing"
        if _is_defect(r.tg_struct, r.tg_struct_base): return "structure"
        return "content" if _is_defect(r.tg_full, r.tg_full_base) else ""

    df["sur_defect"] = df.apply(sur_type, axis=1)
    df["tg_defect"] = df.apply(tg_type, axis=1)
    shape = df.rows_b + df.cols_b + df.span.astype(str)
    df["sur_key"] = np.where(df.sur_defect != "", df.sur_defect + "|" + shape + "|" + df.category, "")
    df["tg_key"] = np.where(df.tg_defect != "", df.tg_defect + "|" + shape + "|" + df.category, "")
    df.to_csv(F("judged.csv"), index=False)
    dv = df[df["base_ok"]]
    print(f"결함 기준: 정답 HTML 대비 TEDS < {tau} (원본 출력과의 비교 없음)")
    print(f"원본이 정답 기준을 통과하지 못한 문서 {len(bad_docs)}건 → 분석 제외 {excl} (입력 {len(df) - len(dv)}개). 분석 대상 입력 {len(dv)}개")
    ok_base = dv[dv["transform"] == "none"]
    print("원본 출력 품질 (표가 검출된 원본):")
    print(ok_base[["tg_struct", "tg_full", "tg_cer_gt"]].describe().round(3).to_string())
    print("결함률 (변형 범주별, 분석 대상만)")
    print(dv.groupby("category")[["sur_defect", "tg_defect"]].agg(lambda s: (s != "").mean()).round(3))
    print("결함 유형 분포 (분석 대상만)")
    print(dv["tg_defect"].replace("", "(정상)").value_counts().to_string())
    print("평균 CER/WER (변형 범주별, 분석 대상만)")
    print(dv.groupby("category")[["mr_cer", "mr_wer", "mr_teds", "tg_cer_gt", "tg_wer_gt"]].agg(["mean", "median"]).round(4).to_string())
    # 6.2절 '선별 기준과 오라클의 중첩': 원본 대비 불일치와 정답 대비 오류의 순위 상관 (원본 입력 제외)
    v = dv[dv["transform"] != "none"]
    rho, p = spearman(v["mr_cer"], v["tg_cer_gt"])
    print(f"중첩도: spearman(MR-CER, 정답 대비 CER) = {rho:.3f} (p={p:.3g}, n={len(v)})")


def _load_cov():
    df = pd.read_csv(F("judged.csv"), keep_default_na=False)
    acts = np.load(W / "acts_pool.npy")
    lo, hi = np.load(W / "prof_lo.npy"), np.load(W / "prof_hi.npy")
    slices = json.loads((W / "layer_slices.json").read_text(encoding="utf-8"))
    pool_ids = list(pd.read_csv(W / "pool.csv", keep_default_na=False)["id"])
    assert pool_ids == list(df["id"]), "pool.csv와 judged.csv 순서 불일치"
    mask = (df["base_ok"].astype(str).str.lower() == "true").to_numpy()
    df, acts = df[mask].reset_index(drop=True), acts[mask]
    print(f"분석 대상 입력 {mask.sum()}개 (원본이 정답 기준 미달인 문서의 입력 {(~mask).sum()}개 제외)")
    return df, build_all(acts, slices, lo, hi, cfg)


def _measure(idx, df, metrics):
    sub = df.iloc[idx]
    out = {"sur_defects": int((sub.sur_defect != "").sum()),
           "tg_defects": int((sub.tg_defect != "").sum()),
           "sur_unique": sub.loc[sub.sur_key != "", "sur_key"].nunique(),
           "tg_unique": sub.loc[sub.tg_key != "", "tg_key"].nunique()}
    for m, (feats, M) in metrics.items():
        out[f"cov_{m}"] = set_coverage(idx, feats, M)
    return out


def stage_rq1(args):
    df, metrics = _load_cov()
    rng = random.Random(f"{cfg.seed}-rq1")
    allidx = list(range(len(df)))
    rows = [_measure(rng.sample(allidx, cfg.rq1_n), df, metrics) for _ in tqdm(range(cfg.rq1_sets), desc="RQ1")]
    r = pd.DataFrame(rows)
    r.to_csv(F("rq1_sets.csv"), index=False)
    out = []
    for m in metrics:
        rho_s, p_s = spearman(r[f"cov_{m}"], r["sur_defects"])
        rho_u, p_u = spearman(r[f"cov_{m}"], r["tg_defects"])
        out.append({"metric": m, "rho_surrogate": rho_s, "p_surrogate": p_s, "rho_target": rho_u, "p_target": p_u})
    out = pd.DataFrame(out)
    out["p_surrogate_holm"] = holm(out.p_surrogate)
    out["p_target_holm"] = holm(out.p_target)
    out.to_csv(F("rq1.csv"), index=False)
    print(f"RQ1 (집합 크기 {cfg.rq1_n} 고정, 무작위 집합 {cfg.rq1_sets}개)")
    print(out.round(3).to_string(index=False))


def stage_select(args):
    df, metrics = _load_cov()
    partition = list(df["partition"])
    mr_cer, mr_teds = df["mr_cer"].to_numpy(float), df["mr_teds"].to_numpy(float)
    allidx = list(range(len(df)))
    rows = []
    for rep in tqdm(range(cfg.n_reps), desc="reps"):
        rng = random.Random(f"{cfg.seed}-rep{rep}")
        sub = rng.sample(allidx, int(len(allidx) * cfg.subpool_ratio))
        for n in cfg.set_sizes:
            if n > len(sub):
                continue
            strat = {"RAND": S.random_sel(sub, n, random.Random(f"{rep}-{n}-rand")),
                     "EP": S.ep_sel(partition, sub, n, random.Random(f"{rep}-{n}-ep")),
                     "MR-CER": S.top_score(mr_cer, sub, n, random.Random(f"{rep}-{n}-mrcer")),
                     "MR-TEDS": S.top_score(mr_teds, sub, n, random.Random(f"{rep}-{n}-mrteds"))}
            for m, (feats, M) in metrics.items():
                strat[f"COV-{m}"] = S.greedy_cov(feats, sub, n, M, random.Random(f"{rep}-{n}-{m}"))
                strat[f"LOW-{m}"] = S.cov_low(feats, sub, n, random.Random(f"{rep}-{n}-low-{m}"))
            for name, idx in strat.items():
                rows.append({"rep": rep, "n": n, "strategy": name, **_measure(idx, df, metrics)})
    pd.DataFrame(rows).to_csv(F("results.csv"), index=False)
    print(f"{len(rows)}행 저장 → {F('results.csv')}")


def _compare(r, a, b, col):
    x, y = r[r.strategy == a][col].values, r[r.strategy == b][col].values
    return {"A": a, "B": b, "mean_A": x.mean(), "mean_B": y.mean(), "A12": a12(x, y), "p": mwu_p(x, y)}


def stage_analyze(args):
    res = pd.read_csv(F("results.csv"))
    col = args.metric   # tg_defects | tg_unique
    out = []
    for n, r in res.groupby("n"):
        others = [s for s in r.strategy.unique() if s != "RAND"]
        comps = [_compare(r, s, "RAND", col) for s in others]                       # RQ2
        cov = [s for s in others if s.startswith("COV-")]
        best = max(cov, key=lambda s: r[r.strategy == s][col].mean())
        for bb in ("MR-CER", "MR-TEDS", "EP"):                                      # RQ3
            comps.append(_compare(r, best, bb, col))
        c = pd.DataFrame(comps)
        c["p_holm"] = holm(c.p)
        c["effect"] = c.A12.map(a12_label)
        c.insert(0, "n", n)
        out.append(c)
    out = pd.concat(out)
    out.to_csv(F(f"analysis_{col}.csv"), index=False)
    print(f"대상 모델: {TGT} / 지표: {col}  (A12 > 0.5 이면 A가, < 0.5 이면 B가 결함을 더 많이 찾음)")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "surrogate", "judge", "rq1", "select", "analyze"])
    ap.add_argument("--tgt", default="docling", choices=["docling", "smoldocling"], help="대상 모델")
    ap.add_argument("--metric", default="tg_defects", choices=["tg_defects", "tg_unique"])
    args = ap.parse_args()
    TGT = args.tgt
    globals()[f"stage_{args.stage}"](args)
