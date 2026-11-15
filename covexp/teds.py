"""TEDS (Tree-Edit-Distance-based Similarity). thead/tbody·th 차이는 정규화로 제거."""
from apted import APTED, Config
from apted.helpers import Tree
from lxml import html as lhtml
from rapidfuzz.distance import Levenshtein


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag, self.colspan, self.rowspan, self.content = tag, colspan, rowspan, content
        self.children = list(children)


class _Cfg(Config):
    def rename(self, a, b):
        if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        if a.tag == "td" and (a.content or b.content):
            return Levenshtein.normalized_distance(a.content, b.content)
        return 0.0


def _span(el, key):
    try:
        return int(el.get(key, "1"))
    except ValueError:
        return 1


def to_tree(html_str: str, structure_only: bool):
    """table → tr → td 3단 트리. 없으면 None."""
    if not html_str or "<table" not in html_str.lower():
        return None
    try:
        root = lhtml.fromstring(html_str)
    except Exception:
        return None
    tables = [root] if root.tag == "table" else root.findall(".//table")
    if not tables:
        return None
    table = tables[0]
    t = TableTree("table")
    for tr in table.iter("tr"):
        r = TableTree("tr")
        for cell in tr:
            if cell.tag not in ("td", "th"):
                continue
            content = [] if structure_only else list(" ".join(cell.text_content().split()))
            r.children.append(TableTree("td", _span(cell, "colspan"), _span(cell, "rowspan"), content))
        t.children.append(r)
    return t


def _count(n):
    return 1 + sum(_count(c) for c in n.children)


def teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    gt = to_tree(gt_html, structure_only)
    pred = to_tree(pred_html, structure_only)
    if gt is None:
        raise ValueError("정답 HTML에 표가 없습니다.")
    if pred is None:
        return 0.0
    dist = APTED(pred, gt, _Cfg()).compute_edit_distance()
    return 1.0 - dist / max(_count(pred), _count(gt))


def table_text(html_str: str) -> str:
    """표 HTML → 셀 텍스트를 행·열 순서대로 이어 붙인 문자열 (CER/WER 계산용)."""
    if not html_str or "<table" not in html_str.lower():
        return ""
    try:
        root = lhtml.fromstring(html_str)
    except Exception:
        return ""
    tables = [root] if root.tag == "table" else root.findall(".//table")
    if not tables:
        return ""
    cells = [" ".join(c.text_content().split()) for tr in tables[0].iter("tr") for c in tr if c.tag in ("td", "th")]
    return " ".join(c for c in cells if c)


def cer(hyp: str, ref: str) -> float:
    """문자 오류율 = (치환 + 삭제 + 삽입) / 기준 문자 수. 1을 넘을 수 있음."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(hyp, ref) / len(ref)


def wer(hyp: str, ref: str) -> float:
    """단어 오류율 (공백 단위). 한국어 어절 단위에는 부적합 — 영문 데이터에만 보고."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(h, r) / len(r)
