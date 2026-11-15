"""문서 이미지 변형. 모두 정답 HTML을 바꾸지 않는(의미 보존) 변형만 포함."""
import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

WHITE = (255, 255, 255)


def rotate(img, sev, rng):
    deg = {1: 1.0, 2: 3.0, 3: 5.0}[sev] * rng.choice([-1, 1])
    return img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=WHITE)


def _persp_coeffs(src, dst):
    # PIL은 출력 좌표(dst) → 입력 좌표(src) 매핑 계수를 받음
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def perspective(img, sev, rng):
    w, h = img.size
    r = {1: 0.02, 2: 0.05, 3: 0.08}[sev]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(x + rng.uniform(-r, r) * w, y + rng.uniform(-r, r) * h) for x, y in src]
    return img.transform((w, h), Image.PERSPECTIVE, _persp_coeffs(src, dst),
                         Image.BICUBIC, fillcolor=WHITE)


def downscale(img, sev, rng):
    f = {1: 0.75, 2: 0.5, 3: 0.35}[sev]
    w, h = img.size
    small = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg(img, sev, rng):
    q = {1: 50, 2: 25, 3: 10}[sev]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def blur(img, sev, rng):
    return img.filter(ImageFilter.GaussianBlur({1: 0.8, 2: 1.5, 3: 2.5}[sev]))


def noise(img, sev, rng):
    sigma = {1: 8, 2: 16, 3: 30}[sev]
    a = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randrange(2**32))
    a = np.clip(a + nrng.normal(0, sigma, a.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def stain(img, sev, rng):
    w, h = img.size
    out = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range({1: 2, 2: 5, 3: 10}[sev]):
        rx, ry = rng.uniform(0.03, 0.12) * w, rng.uniform(0.03, 0.12) * h
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        g = rng.randint(120, 200)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(g, g - 10, g - 30, rng.randint(60, 120)))
    return Image.alpha_composite(out, layer).convert("RGB")


def watermark(img, sev, rng):
    w, h = img.size
    alpha = {1: 40, 2: 80, 3: 120}[sev]
    size = max(16, min(w, h) // 5)
    try:
        font = ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((w * 0.1, h * 0.35), "CONFIDENTIAL", font=font, fill=(200, 0, 0, alpha))
    layer = layer.rotate(rng.uniform(15, 35), resample=Image.BICUBIC)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def remove_lines(img, sev, rng):
    """표 괘선을 검출해 일부(강도별 33/66/100%)를 지운다."""
    gray = np.asarray(img.convert("L"))
    h, w = gray.shape
    dark = gray < 128
    horiz = ndimage.binary_opening(dark, structure=np.ones((1, max(10, w // 15))))
    vert = ndimage.binary_opening(dark, structure=np.ones((max(10, h // 15), 1)))
    lines = horiz | vert
    lab, n = ndimage.label(lines)
    if n == 0:
        return img.copy()
    frac = {1: 0.33, 2: 0.66, 3: 1.0}[sev]
    ids = list(range(1, n + 1))
    rng.shuffle(ids)
    kill = np.isin(lab, ids[: max(1, round(n * frac))])
    kill = ndimage.binary_dilation(kill, iterations=1)
    a = np.asarray(img).copy()
    a[kill] = 255
    return Image.fromarray(a)


TRANSFORMS = {
    # 이름: (범주, 함수)
    "rotate": ("geometric", rotate),
    "perspective": ("geometric", perspective),
    "downscale": ("quality", downscale),
    "jpeg": ("quality", jpeg),
    "blur": ("quality", blur),
    "noise": ("noise", noise),
    "stain": ("noise", stain),
    "watermark": ("noise", watermark),
    "remove_lines": ("structure", remove_lines),
}


def apply(name, img, sev, seed_str):
    rng = random.Random(seed_str)   # 문자열 시드: 실행 간 재현 가능
    return TRANSFORMS[name][1](img.convert("RGB"), sev, rng)
