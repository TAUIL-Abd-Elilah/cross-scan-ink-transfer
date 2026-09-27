"""Blur control for the letter-scale score: rescore reads after blurring EVERY output by sigma 0/2/4 px, on
identical pixels (intersection of the models' pred>0 masks), so a gain that is only smoothing shows up."""
import json, os, sys
import numpy as np, tifffile, zarr
from scipy.ndimage import gaussian_filter
from hp_eval import OUT, hp, r, SMO, SIG
models = ['ink9um_s42_075k', 'ink9um_s43_075k', 'ft_b_6k', 'ft_a_4k']
res = {}
for seg in sys.argv[1:]:
    d = os.path.join(OUT, seg)
    sv = np.asarray(zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')[14])
    key = np.load(os.path.join(d, 'canon_al.npy')).astype(np.float32)
    preds = {}
    for m in models:
        p = tifffile.imread(os.path.join(d, f'pred_{m}.tif')).astype(np.float32)
        preds[m] = p / 255 if p.max() > 1.5 else p
    H = min([sv.shape[0], key.shape[0]] + [p.shape[0] for p in preds.values()])
    W = min([sv.shape[1], key.shape[1]] + [p.shape[1] for p in preds.values()])
    valid = (sv[:H, :W] > 0) & np.isfinite(key[:H, :W])
    k = np.where(valid, key[:H, :W], 0)
    core = valid & (gaussian_filter(valid.astype(np.float32), 2 * SIG) > 0.999)
    m = core.copy()
    for p in preds.values():
        m &= p[:H, :W] > 0
    kh = hp(k, m)
    for b in (0, 2, 4):
        for name, p in preds.items():
            q = p[:H, :W] if b == 0 else gaussian_filter(p[:H, :W], b)
            v = r(hp(q, m)[m], kh[m])
            res[f'{seg}|{name}|blur{b}'] = v
            print(f'{seg[-26:]:26s} blur {b} {name:16s} hp r {v:+.3f}', flush=True)
json.dump(res, open('results_hp_blur.json', 'w'), indent=1)
