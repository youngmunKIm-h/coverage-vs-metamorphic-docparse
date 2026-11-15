"""PubTabNet(11.24GB tar.gz)을 디스크에 저장하지 않고 스트리밍으로 열어 필요한 파일만 추출.
추출: 주석 jsonl, val 이미지 전체, train 이미지 약 1/500 (파일명 해시로 결정적 선택, 약 1,000장)
  python fetch_pubtabnet.py            # Hugging Face에서 스트리밍
  python fetch_pubtabnet.py 경로.tar.gz  # 이미 받은 파일이 있으면 그 파일에서 추출
중단되면 처음부터 다시 읽지만, 이미 추출된 파일은 다시 쓰지 않음.
"""
import hashlib
import shutil
import sys
import tarfile
import time
from pathlib import Path

OUT = Path("data/pubtabnet")
SRC = "datasets/ajimeno/PubTabNet/pubtabnet.tar.gz"
TOTAL_GB = 11.24


def keep_train(name: str) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 500 == 0


def target_of(member_name: str):
    parts = Path(member_name).parts
    name = parts[-1]
    if name.endswith(".jsonl"):
        return OUT / name
    if "val" in parts:
        return OUT / "val" / name
    if "train" in parts and keep_train(name):
        return OUT / "train" / name
    return None


def run(fileobj):
    t0, n_seen, n_saved, last = time.time(), 0, 0, 0.0
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            n_seen += 1
            dst = target_of(m.name)
            if dst is not None:
                if not (dst.exists() and dst.stat().st_size == m.size):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as r, open(dst, "wb") as w:
                        shutil.copyfileobj(r, w)
                n_saved += 1
            now = time.time()
            if now - last > 30:
                last = now
                try:
                    gb = fileobj.tell() / 1e9
                    prog = f"{gb:.2f}/{TOTAL_GB} GB ({gb / TOTAL_GB:.0%})"
                except Exception:
                    prog = "?"
                print(f"  {prog}  확인 {n_seen:,}개 / 추출 {n_saved:,}개  경과 {(now - t0) / 60:.1f}분", flush=True)
    print(f"완료: 확인 {n_seen:,}개, 추출 {n_saved:,}개, {(time.time() - t0) / 60:.1f}분")
    for sub in ("train", "val"):
        d = OUT / sub
        print(f"  {d}: {len(list(d.glob('*'))) if d.exists() else 0}개")
    print("  jsonl:", [p.name for p in OUT.glob("*.jsonl")])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            run(f)
    else:
        from huggingface_hub import HfFileSystem
        with HfFileSystem().open(SRC, "rb", block_size=8 * 2**20) as f:
            run(f)
