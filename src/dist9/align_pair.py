"""Measure the grid offset between the native 9.362 um render and the downsampled 2.4 um labels.

Runs ink_9um (hybrid_3d2d seed42 step-075000, forward) on sv.zarr, then an NCC search over
integer offsets (+-R px at 9.362 um) of that prediction against canon.npy. Writes
  ink9um_s42_075000.tif, align.json {offset_yx, r_best, r_zero, r_null_p99}
and canon_al.npy / human_al.npy shifted so that label[y, x] sits under sv[:, y, x].
"""
import json
import os
import subprocess
import sys

import numpy as np
import tifffile

PY = sys.executable
INK9UM_SRC = os.environ.get('INK9UM_SRC', 'villa/vesuvius/src')      # villa checkout with vesuvius.ink_detection
CKPTS = os.environ.get('INK9UM_CKPTS', 'models/ink_9um')           # huggingface.co/scrollprize/ink_9um
ENV = dict(os.environ, PYTHONPATH=INK9UM_SRC)
OUT = os.environ.get('DIST9_DATA', 'dist9_data')
R = 24


def infer(d, ckpt, name, extra=()):
    out = os.path.join(d, name + '.tif')
    if not os.path.exists(out):
        subprocess.check_call([PY, '-m', 'vesuvius.ink_detection.inference.infer', os.path.join(d, 'sv.zarr'), ckpt, out,
                               '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'forward',
                               '--no-compile', *extra], env=ENV, stdout=open(os.path.join(d, name + '.log'), 'w'),
                              stderr=subprocess.STDOUT)
    a = tifffile.imread(out).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def shift(a, dy, dx, fill=np.nan):
    out = np.full_like(a, fill)
    H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def ncc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def stretched(d, H, W):
    """2.4 um maps resized onto exactly the native (H, W) canvas: the two canvases cover the same
    extent but differ by ~0.3-0.4 % from a pure 9.362/2.399 scale (measured on w030: quadrant
    offsets drift ~11 px top to bottom), so a shift alone cannot align them."""
    import cv2
    import tifffile
    import zarr
    p = tifffile.imread(os.path.join(d, 'canon_2p4.tif'))
    if p.ndim == 3:
        p = p[..., 0] if p.shape[-1] in (1, 3, 4) else p[0]
    canon = cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    del p
    human = None
    lz = os.path.join(d, 'inklabels.zarr', '0')
    if os.path.exists(lz):
        lab = ((zarr.open_array(lz, mode='r')[:] > 0) * 255).astype(np.uint8)
        human = cv2.resize(lab, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return canon, human


def highpass(a, s=16.0):
    from scipy import ndimage
    b = np.nan_to_num(a)
    return b - ndimage.gaussian_filter(b, s)


def main(seg):
    d = os.path.join(OUT, seg)
    pred = infer(d, f'{CKPTS}/hybrid_3d2d-seed42/step-075000.pth', 'ink9um_s42_075000')
    sv = np.asarray(__import__('zarr').open_array(os.path.join(d, 'sv.zarr'), mode='r')[14])
    H, W = sv.shape
    canon, human = stretched(d, H, W)
    pred = pred[:H, :W]
    valid = sv > 0
    pr = np.where(valid, highpass(np.where(valid, pred, 0)), np.nan)
    ca = highpass(canon)
    q = 2                                              # search on a 2x-downsampled grid, refine at full
    prq, caq = pr[::q, ::q], ca[::q, ::q]
    S = {}
    for dy in range(-R // q, R // q + 1):
        for dx in range(-R // q, R // q + 1):
            S[(dy, dx)] = ncc(prq, shift(caq, dy, dx))
    (by, bx) = max((k for k in S if np.isfinite(S[k])), key=lambda k: S[k])
    best = (-2.0, 0, 0)
    for dy in range(by * q - 2, by * q + 3):
        for dx in range(bx * q - 2, bx * q + 3):
            r = ncc(pr[::2, ::2], shift(ca, dy, dx)[::2, ::2])
            if np.isfinite(r) and r > best[0]:
                best = (r, dy, dx)
    vals = np.array([v for v in S.values() if np.isfinite(v)])
    r0 = ncc(pr[::2, ::2], ca[::2, ::2])
    rb, dy, dx = best
    # per-quadrant offsets: a scale mismatch shows as offsets that grow across the segment
    quads = {}
    for qn, (ys, xs) in {'tl': (slice(0, H // 2), slice(0, W // 2)), 'tr': (slice(0, H // 2), slice(W // 2, W)),
                         'bl': (slice(H // 2, H), slice(0, W // 2)), 'br': (slice(H // 2, H), slice(W // 2, W))}.items():
        bq = (-2.0, 0, 0)
        for ddy in range(dy - 12, dy + 13, 2):
            for ddx in range(dx - 12, dx + 13, 2):
                r = ncc(pr[ys, xs][::2, ::2], shift(ca, ddy, ddx)[ys, xs][::2, ::2])
                if np.isfinite(r) and r > bq[0]:
                    bq = (r, ddy, ddx)
        quads[qn] = {'r': round(bq[0], 3), 'offset_yx': [bq[1], bq[2]]}
    out = {'quadrants': quads, 'segment': seg, 'offset_yx': [dy, dx], 'r_best': rb, 'r_zero_offset': r0,
           'r_search_p50': float(np.percentile(vals, 50)), 'r_search_p99': float(np.percentile(vals, 99)),
           'aligner': 'ink_9um hybrid_3d2d-seed42 step-075000 forward, high-passed (sigma 16), canon stretched to the native canvas'}
    json.dump(out, open(os.path.join(d, 'align.json'), 'w'), indent=1)
    np.save(os.path.join(d, 'canon_al.npy'), shift(np.where(valid, canon, np.nan), dy, dx).astype(np.float16))
    if human is not None:
        np.save(os.path.join(d, 'human_al.npy'), shift(human, dy, dx, 0.0).astype(np.float16))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        main(s)
