"""Human-label benchmark for reading the MARCH-2026 2.4 um scan (PHerc0139 20260319133554) with the canonical model.

For each PHerc0139 segment that has human ink labels (drawn on the 2.399 um scan 20260102150214) and a mesh on
the March scan:
  1. pick the densest labelled window (<= WIN_H x WIN_W grid cells of the 2.399 canvas, 20 px per cell);
  2. REFERENCE: fetch that window from the published 2.399 um surface volume and run the canonical model
     (centre 62-layer window) -> AUC vs labels;
  3. MARCH: map the window onto the March mesh grid (valid-mask offset search), render it from the March
     scan with 4.8 um data on the 2.4 um grid (161 layers), then
       a. mesh as published: centre window;
       b. flattened onto the local layering + depth search over -40..+40 layers (5 windows);
     AUC vs labels for each; report the mesh-as-is AUC and the best-depth AUC.
Label/pred alignment is searched (+-48 ds8 px), as in label_eval.py. Background = labelled-ink-free pixels
within 2 mm of labelled ink.
usage: march_bench.py <segment> [<segment> ...]
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import tifffile
import zarr
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
OUT = os.environ.get('MARCH_BENCH_DATA', 'out/march_bench')
VOL_MARCH = f'{B}/PHerc0139/volumes/20260319133554-2.403um-0.2m-77keV-masked.zarr'
WIN_H, WIN_W = 60, 170          # grid cells (x20 px): 1200 x 3400 px = 2.9 x 8.2 mm
DS = 8


def get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def mesh(seg, scan, d):
    for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
        get(f'{B}/PHerc0139/segments/{seg}/mesh/{seg.split("-")[0]}-on-{scan}.tifxyz/{f}', os.path.join(d, f))
    x = tifffile.imread(os.path.join(d, 'x.tif'))
    return x > -0.5


def labels(seg, d):
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix=PHerc0139/segments/{seg}/ink-labels/2.399um-volume-20260102150214/').read().decode()
    import re
    lz = sorted(set(re.findall(r'<Key>([^<]*inklabels\.zarr)/0/zarr\.json</Key>', lst)))[-1]
    root = os.path.join(d, 'inklabels.zarr')
    get(f'{B}/{lz}/zarr.json', os.path.join(root, 'zarr.json'))
    get(f'{B}/{lz}/0/zarr.json', os.path.join(root, '0', 'zarr.json'))
    get(f'{B}/{lz}/0/c/0/0', os.path.join(root, '0', 'c', '0', '0'))
    return zarr.open_array(os.path.join(root, '0'), mode='r')[:] > 0


def auc(score, pos, neg):
    sp, sn = score[pos], score[neg]
    if sp.size < 50 or sn.size < 50:
        return float('nan')
    rng = np.random.default_rng(0)
    sp = rng.choice(sp, min(sp.size, 200000), replace=False); sn = rng.choice(sn, min(sn.size, 200000), replace=False)
    allv = np.concatenate([sp, sn]); rk = allv.argsort(kind='mergesort').argsort() + 1
    return float((rk[:sp.size].sum() - sp.size * (sp.size + 1) / 2) / (sp.size * sn.size))


def down(a, ds=DS):
    h, w = a.shape[0] // ds * ds, a.shape[1] // ds * ds
    return a[:h, :w].reshape(h // ds, ds, w // ds, ds).mean((1, 3), dtype=np.float32)


def eval_map(ink, lab8, r0c0, R=48):
    """ink: full-res map of the window; lab8: ds8 labels of the whole canvas; r0c0: expected ds8 origin.
    Search offset maximising label NCC, then AUC (pos: lab>0.5; neg: lab<0.05 within 2 mm of ink)."""
    t = down(ink)
    h, w = t.shape
    ey, ex = r0c0
    best = (-2, 0, 0)
    for dy in range(-R, R + 1, 2):
        for dx in range(-R, R + 1, 2):
            y, x = ey + dy, ex + dx
            if y < 0 or x < 0 or y + h > lab8.shape[0] or x + w > lab8.shape[1]:
                continue
            L = lab8[y:y + h, x:x + w]
            if L.std() == 0:
                continue
            a, b = t - t.mean(), L - L.mean()
            r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
            if r > best[0]:
                best = (r, dy, dx)
    r, dy, dx = best
    L = lab8[ey + dy:ey + dy + h, ex + dx:ex + dx + w]
    near = ndimage.distance_transform_edt(L < 0.5) <= 2.0 / (2.399e-3 * DS)
    return {'auc': auc(t, L >= 0.5, near & (L < 0.05)), 'r': r, 'offset': [dy, dx], 'label_frac': float((L >= 0.5).mean())}


def run(seg):
    d = os.path.join(OUT, seg)
    os.makedirs(d, exist_ok=True)
    rp = os.path.join(d, 'result.json')
    if os.path.exists(rp):
        return json.load(open(rp))
    lab = labels(seg, d)
    va = mesh(seg, '20260319133554-2.403um', os.path.join(d, 'mesh_march'))
    vb = mesh(seg, '20260102150214-2.399um', os.path.join(d, 'mesh_2399'))
    # densest labelled window on the 2.399 grid (cells of 20 px)
    Hc, Wc = lab.shape[0] // 20, lab.shape[1] // 20
    cells = lab[:Hc * 20, :Wc * 20].reshape(Hc, 20, Wc, 20).mean((1, 3), dtype=np.float32)
    dens = ndimage.uniform_filter(cells, (WIN_H, WIN_W), mode='constant')
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    r0, c0 = int(np.clip(cy - WIN_H // 2, 0, Hc - WIN_H)), int(np.clip(cx - WIN_W // 2, 0, Wc - WIN_W))
    # grid offset march vs 2.399 mesh (valid masks)
    A, Bm = va.astype(np.float32), vb.astype(np.float32)
    best = (-1, 0, 0)
    Hh, Ww = min(A.shape[0], Bm.shape[0]) - 20, min(A.shape[1], Bm.shape[1]) - 20
    for dy in range(-10, 11):
        for dx in range(-10, 11):
            s = (A[10 + dy:10 + dy + Hh, 10 + dx:10 + dx + Ww] == Bm[10:10 + Hh, 10:10 + Ww]).mean()
            if s > best[0]:
                best = (s, dy, dx)
    _, gdy, gdx = best
    lab8 = down(lab)
    e0 = (r0 * 20 // DS, c0 * 20 // DS)
    res = {'segment': seg, 'window_cells': [r0, r0 + WIN_H, c0, c0 + WIN_W], 'mesh_grid_offset': [gdy, gdx],
           'window_label_frac': float(cells[r0:r0 + WIN_H, c0:c0 + WIN_W].mean())}
    # REFERENCE: published 2.399 surface volume
    ref = os.path.join(d, 'ref')
    if not os.path.exists(ref + '.npy'):
        subprocess.check_call([PY, os.path.join(HERE, 'fetch_sv.py'),
                               f'{B}/PHerc0139/segments/{seg}/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr',
                               ref, str(r0 * 20), str((r0 + WIN_H) * 20), str(c0 * 20), str((c0 + WIN_W) * 20)])
    json.dump({'rows': [r0, r0 + WIN_H], 'cols': [c0, c0 + WIN_W], 'upsample': 20, 'grid': 'published'}, open(ref + '.json', 'w'))
    sv = np.load(ref + '.npy', mmap_mode='r')
    c = (sv.shape[0] - 1) // 2
    if not os.path.exists(ref + '_ink_o+0.npy'):
        subprocess.check_call([PY, os.path.join(HERE, 'infer_crop.py'), ref, '--offsets', '0'], stdout=subprocess.DEVNULL)
    res['reference_2399'] = eval_map(np.load(ref + '_ink_o+0.npy'), lab8, e0)
    # MARCH: render from the March scan on the March mesh grid
    mar = os.path.join(d, 'march')
    if not os.path.exists(mar + '.npy'):
        subprocess.check_call([PY, os.path.join(HERE, 'render_crop.py'), os.path.join(d, 'mesh_march'), VOL_MARCH, mar,
                               '--rows', str(r0 + gdy), str(r0 + gdy + WIN_H), '--cols', str(c0 + gdx), str(c0 + gdx + WIN_W),
                               '--layers', '161', '--level', '1', '--oversample', '2', '--smooth-normals', '1.0',
                               '--cache-chunks', '1500'], stdout=subprocess.DEVNULL)
    if not os.path.exists(mar + '_ink_o+0.npy'):
        subprocess.check_call([PY, os.path.join(HERE, 'infer_crop.py'), mar, '--offsets', '0'], stdout=subprocess.DEVNULL)
    res['march_mesh_as_is'] = eval_map(np.load(mar + '_ink_o+0.npy'), lab8, e0)
    if not os.path.exists(mar + '_flat.npy'):
        subprocess.check_call([PY, os.path.join(HERE, 'refine_surface.py'), mar, '--out-layers', '145'], stdout=subprocess.DEVNULL)
    offs = [-40, -20, 0, 20, 40]
    need = [o for o in offs if not os.path.exists(mar + f'_flat_ink_o{o:+d}.npy')]
    if need:
        subprocess.check_call([PY, os.path.join(HERE, 'infer_crop.py'), mar + '_flat', '--offsets'] + [str(o) for o in need],
                              stdout=subprocess.DEVNULL)
    res['march_flat_depths'] = {str(o): eval_map(np.load(mar + f'_flat_ink_o{o:+d}.npy'), lab8, e0) for o in offs}
    bo = max(res['march_flat_depths'], key=lambda k: res['march_flat_depths'][k]['auc'])
    res['march_best_depth'] = dict(res['march_flat_depths'][bo], depth=int(bo))
    json.dump(res, open(rp, 'w'), indent=1)
    for f in ('march.npy', 'march_flat.npy'):                   # keep ink maps, drop the big stacks
        p = os.path.join(d, f)
        if os.path.exists(p):
            os.remove(p)
    return res


if __name__ == '__main__':
    for s in sys.argv[1:]:
        r = run(s)
        print(s, 'ref', round(r['reference_2399']['auc'], 3), 'march as-is', round(r['march_mesh_as_is']['auc'], 3),
              'march best', round(r['march_best_depth']['auc'], 3), 'depth', r['march_best_depth']['depth'], flush=True)
