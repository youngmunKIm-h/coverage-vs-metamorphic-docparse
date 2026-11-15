"""화이트박스 대리 모델: Table Transformer(표 구조 인식) + 뉴런 활성 후킹."""
import numpy as np
import torch
from torch import nn
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Surrogate:
    """뉴런 정의
    - CNN 백본: nn.ReLU 출력의 채널별 공간 평균 (채널 1개 = 뉴런 1개)
    - 트랜스포머: 인코더/디코더 FFN fc1 출력에 ReLU 적용 후 토큰 평균 (차원 1개 = 뉴런 1개)
    배치 크기 1로만 실행 (패딩 토큰이 평균에 섞이지 않게).
    """

    def __init__(self, model_name, threshold=0.5, max_side=1000, device=None):
        from transformers import TableTransformerForObjectDetection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.id2label = self.model.config.id2label
        self.threshold, self.max_side = threshold, max_side
        self._buf = {}
        self.layer_names = []
        for name, m in self.model.named_modules():
            is_relu = isinstance(m, nn.ReLU) and ".backbone." in f".{name}."
            is_fc1 = isinstance(m, nn.Linear) and name.endswith("fc1")
            if is_relu or is_fc1:
                m.register_forward_hook(self._hook(name, is_fc1))
                self.layer_names.append(name)
        if not self.layer_names:
            raise RuntimeError("후킹할 레이어를 찾지 못했습니다. model.named_modules()를 출력해 확인하세요.")
        self.layer_slices = None

    def _hook(self, name, is_fc1):
        def fn(mod, inp, out):
            a = out.detach()
            if is_fc1:
                a = torch.relu(a)
            if a.dim() == 4:          # (B, C, H, W)
                a = a.mean(dim=(2, 3))
            elif a.dim() == 3:        # (B, T, C)
                a = a.mean(dim=1)
            self._buf[name] = a[0].float().cpu()
        return fn

    def _prep(self, img: Image.Image):
        img = img.convert("RGB")
        w, h = img.size
        s = self.max_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return ((x - MEAN) / STD).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def run(self, img: Image.Image):
        """반환: (활성 벡터 np.float32[N], 검출 결과 list)"""
        self._buf.clear()
        out = self.model(pixel_values=self._prep(img))
        missing = [n for n in self.layer_names if n not in self._buf]
        if missing:
            raise RuntimeError(f"forward 중 호출되지 않은 레이어: {missing[:5]}")
        vecs = [self._buf[n].numpy() for n in self.layer_names]
        if self.layer_slices is None:
            ofs, self.layer_slices = 0, []
            for v in vecs:
                self.layer_slices.append((ofs, ofs + len(v)))
                ofs += len(v)
        acts = np.concatenate(vecs).astype(np.float32)

        W, H = img.size
        probs = out.logits.softmax(-1)[0, :, :-1]      # 마지막 클래스 = no-object
        scores, labels = probs.max(-1)
        boxes = out.pred_boxes[0]                      # (cx, cy, w, h), 0~1 정규화
        dets = []
        for sc, lb, b in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
            if sc < self.threshold:
                continue
            cx, cy, bw, bh = b
            dets.append({"label": self.id2label[lb], "score": sc,
                         "box": [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]})
        return acts, dets


def _ov(a, b, axis):
    """a 박스 길이 대비 a∩b 겹침 비율 (axis 0 = x, 1 = y)."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    inter = max(0.0, min(a[hi], b[hi]) - max(a[lo], b[lo]))
    return inter / max(1e-6, a[hi] - a[lo])


def _dedup(items, axis):
    kept = []
    for it in items:
        if not kept or _ov(it["box"], kept[-1]["box"], axis) < 0.5:
            kept.append(it)
    return kept


def dets_to_html(dets) -> str:
    """행·열·병합 셀 검출 → 셀 텍스트 없는 HTML 구조 (구조 TEDS 전용)."""
    ctr = lambda d, ax: (d["box"][ax] + d["box"][ax + 2]) / 2
    rows = _dedup(sorted([d for d in dets if d["label"] == "table row"], key=lambda d: ctr(d, 1)), 1)
    cols = _dedup(sorted([d for d in dets if d["label"] == "table column"], key=lambda d: ctr(d, 0)), 0)
    if not rows or not cols:
        return ""
    R, C = len(rows), len(cols)
    owner = [[None] * C for _ in range(R)]
    anchors = {}
    for k, s in enumerate(d for d in dets if d["label"] == "table spanning cell"):
        rr = [i for i, r in enumerate(rows) if _ov(r["box"], s["box"], 1) > 0.5]
        cc = [j for j, c in enumerate(cols) if _ov(c["box"], s["box"], 0) > 0.5]
        if len(rr) * len(cc) <= 1:
            continue
        if rr != list(range(rr[0], rr[-1] + 1)) or cc != list(range(cc[0], cc[-1] + 1)):
            continue
        if any(owner[i][j] is not None for i in rr for j in cc):
            continue
        for i in rr:
            for j in cc:
                owner[i][j] = k
        anchors[(rr[0], cc[0])] = (len(rr), len(cc))
    out = ["<table>"]
    for i in range(R):
        out.append("<tr>")
        for j in range(C):
            if (i, j) in anchors:
                rs, cs = anchors[(i, j)]
                attr = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
                out.append(f"<td{attr}></td>")
            elif owner[i][j] is None:
                out.append("<td></td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
