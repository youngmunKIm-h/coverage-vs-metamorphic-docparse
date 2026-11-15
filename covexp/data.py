"""PubTabNet 로딩, 정답 HTML 생성, 동등 클래스 특성 계산."""
import json
import random
from html import escape
from pathlib import Path


def reservoir_sample(jsonl: Path, split: str, n: int, seed: int, img_dir: Path = None):
    """대용량 jsonl을 메모리에 다 올리지 않고 split별로 n개를 무작위 추출.
    img_dir를 주면 그 폴더에 이미지가 실제로 있는 항목만 대상으로 함 (부분 추출 대응)."""
    rng = random.Random(seed)
    sample, seen = [], 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a["split"] != split:
                continue
            if img_dir is not None and not (img_dir / a["filename"]).exists():
                continue
            seen += 1
            if len(sample) < n:
                sample.append(a)
            else:
                j = rng.randrange(seen)
                if j < n:
                    sample[j] = a
    if len(sample) < n:
        raise ValueError(f"{split} 분할 표본이 {len(sample)}개뿐입니다 (요청 {n}).")
    return sample


def gt_html(ann) -> str:
    """PubTabNet 구조 토큰 + 셀 토큰 → 정답 HTML (PubTabNet 참조 구현 방식)."""
    tokens = list(ann["html"]["structure"]["tokens"])
    cells = ann["html"].get("cells", ann["html"].get("cell"))
    idx = [i for i, t in enumerate(tokens) if t in ("<td>", ">")]
    for i, cell in zip(idx[::-1], cells[::-1]):
        if cell["tokens"]:
            text = "".join(escape(t) if len(t) == 1 else t for t in cell["tokens"])
            tokens.insert(i + 1, text)
    return "<table>" + "".join(tokens) + "</table>"


def table_features(ann) -> dict:
    """동등 클래스 분할용 표 특성: 행 수·열 수 구간, 병합 셀 유무."""
    toks = ann["html"]["structure"]["tokens"]
    n_rows = toks.count("<tr>")
    first_row, in_first = 0, False
    for t in toks:
        if t == "<tr>":
            if in_first:
                break
            in_first = True
        elif in_first and t in ("<td>", "<td"):
            first_row += 1
    has_span = any(("colspan" in t) or ("rowspan" in t) for t in toks)
    rows_b = "R1" if n_rows <= 5 else ("R2" if n_rows <= 15 else "R3")
    cols_b = "C1" if first_row <= 4 else ("C2" if first_row <= 8 else "C3")
    return {"rows_b": rows_b, "cols_b": cols_b, "span": int(has_span)}
