"""대상 파서 실행 (별도 conda 환경 covtarget). 입력과 출력만 사용하는 블랙박스로 취급.
  python run_target.py docling            # Docling (TableFormer 파이프라인)
  python run_target.py smoldocling        # SmolDocling-256M (종단간 VLM)
  python run_target.py docling --limit 5  # 설치 확인용 소량 실행
결과: work/target_pred_<name>.json  (입력 id → 표 HTML)
원시 로그: work/target_raw_<name>.jsonl (중단 후 재실행하면 이어서 처리)
"""
import argparse
import csv
import json
import time
from pathlib import Path

W = Path("work")


def load_ids():
    with open(W / "pool.csv", encoding="utf-8", newline="") as f:
        return [r["id"] for r in csv.DictReader(f)]


def _table_html(t, doc):
    try:
        return t.export_to_html(doc=doc)
    except TypeError:          # 구버전 docling-core
        return t.export_to_html()


def pick_table(htmls):
    htmls = [h for h in htmls if h and "<table" in h.lower()]
    return max(htmls, key=len) if htmls else ""


class DoclingTarget:
    def __init__(self):
        from docling.document_converter import DocumentConverter
        self.conv = DocumentConverter()     # 이미지 입력 시 OCR 포함 기본 파이프라인

    def __call__(self, path):
        doc = self.conv.convert(str(path)).document
        return pick_table([_table_html(t, doc) for t in doc.tables]), None


try:
    from transformers import StoppingCriteria as _StopBase
except Exception:          # transformers 없는 환경에서도 docling 대상은 동작하도록
    _StopBase = object


class _DegenerationStop(_StopBase):
    """퇴화(같은 토큰 반복) 감지: 생성 토큰의 최근 window개 중 서로 다른 토큰이 max_distinct개 이하면 중단.
    정상 표 출력에는 숫자·텍스트·위치 토큰이 섞이므로 걸리지 않음. 중단된 출력도 그대로 판정에 사용."""

    def __init__(self, prompt_len, window=128, max_distinct=4):
        self.prompt_len, self.window, self.max_distinct = prompt_len, window, max_distinct
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        gen = input_ids[0, self.prompt_len:]
        hit = gen.shape[0] >= self.window and torch.unique(gen[-self.window:]).numel() <= self.max_distinct
        self.triggered = self.triggered or bool(hit)
        return torch.full((input_ids.shape[0],), bool(hit), dtype=torch.bool, device=input_ids.device)


class SmolDoclingTarget:
    MODEL = "ds4sd/SmolDocling-256M-preview"   # 저장소 이름이 바뀌었으면 여기만 수정

    def __init__(self, max_new_tokens=4096):
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoProcessor
        try:                                         # 최신 transformers
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:                          # 구버전
            from transformers import AutoModelForVision2Seq as AutoVLM
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.dev == "cuda" else torch.float32
        # transformers 4.56부터 torch_dtype → dtype 으로 인자 이름 변경
        key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        self.proc = AutoProcessor.from_pretrained(self.MODEL)
        self.model = AutoVLM.from_pretrained(self.MODEL, **{key: dtype}).to(self.dev).eval()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}]
        self.prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        self.max_new_tokens = max_new_tokens
        self.last_meta = {}

    def __call__(self, path):
        from PIL import Image
        from transformers import StoppingCriteriaList
        from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
        img = Image.open(path).convert("RGB")
        inputs = self.proc(text=self.prompt, images=[img], return_tensors="pt").to(self.dev)
        plen = inputs["input_ids"].shape[1]
        stop = _DegenerationStop(plen)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      stopping_criteria=StoppingCriteriaList([stop]))
        self.last_meta = {"ntok": int(out.shape[1] - plen), "degenerate": stop.triggered}
        tags = self.proc.batch_decode(out[:, plen:], skip_special_tokens=False)[0].lstrip()
        dt = DocTagsDocument.from_doctags_and_image_pairs([tags], [img])
        doc = DoclingDocument.load_from_doctags(dt, document_name="page")
        return pick_table([_table_html(t, doc) for t in doc.tables]), tags


TARGETS = {"docling": DoclingTarget, "smoldocling": SmolDoclingTarget}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(TARGETS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = load_ids()
    if args.limit:
        ids = ids[: args.limit]
    log = W / f"target_raw_{args.name}.jsonl"
    done, errors = {}, {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("error"):
                errors[r["id"]] = r["error"]
            else:
                done[r["id"]] = r["html"]
                errors.pop(r["id"], None)
    todo = [i for i in ids if i not in done]
    print(f"[{args.name}] 전체 {len(ids)} / 완료 {len(ids) - len(todo)} / 남음 {len(todo)}")

    target = TARGETS[args.name]()
    t_start, streak = time.perf_counter(), 0
    with open(log, "a", encoding="utf-8") as f:
        for k, pid in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                html, raw = target(W / "pool" / f"{pid}.png")
                err = None
                streak = 0
            except Exception as e:      # 오류는 결함으로 세지 않고 별도 기록
                html, raw, err = "", None, f"{type(e).__name__}: {e}"
                streak += 1
                print(f"[오류] {pid}: {err}")
                if streak >= 3 and k == streak:
                    raise RuntimeError("처음 3건이 연속 실패했습니다. 설치·모델 로딩 문제일 가능성이 큽니다. 위 오류 메시지를 확인하세요.")
            rec = {"id": pid, "html": html, "sec": round(time.perf_counter() - t0, 3), "error": err, "raw": raw,
                   **(getattr(target, "last_meta", None) or {})}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if err:
                errors[pid] = err
            else:
                done[pid] = html
                errors.pop(pid, None)
            if k % 50 == 0 or k == len(todo):
                el = time.perf_counter() - t_start
                print(f"  {k}/{len(todo)}  평균 {el / k:.2f}초/장  예상 잔여 {(len(todo) - k) * el / k / 60:.1f}분")

    preds = {i: done[i] for i in ids if i in done}
    (W / f"target_pred_{args.name}.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {len(preds)}건, 표 미검출 {sum(v == '' for v in preds.values())}건, 실행 오류 {len(errors)}건")
    if args.name == "smoldocling":
        recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        recs = {r["id"]: r for r in recs if not r.get("error")}
        deg = sum(1 for i in ids if recs.get(i, {}).get("degenerate"))
        print(f"퇴화 감지로 중단된 출력: {deg}건 (표 미검출·구조 오류 결함으로 판정됨)")
    if errors:
        print("실행 오류 입력은 결과에서 빠졌습니다. 재실행하면 오류 건만 다시 시도합니다.")


if __name__ == "__main__":
    main()
