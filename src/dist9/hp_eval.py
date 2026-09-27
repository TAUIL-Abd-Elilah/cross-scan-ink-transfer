"""Letter-scale score (Scheirer's hp r, ShribyrLabs/vesuvius-reports report 02, MIT) for our 9 um reads.

hp r = Pearson r between read and key after both are high-passed with a normalised Gaussian of sigma 48 um
(20/3.9 px at 9.362 um). "inked hp r" restricts to the key's inked area (key smoothed sigma ~100 um > 60/255).
Pixels: key valid, sv valid, pred > 0, and away from the border (normalised-support > 0.999 at 2 sigma).
Null: max |r| with the key rolled by 150, 300 and -300 rows (his choice).
Keys: canonical 2.4 um prediction on the native grid (canon_al.npy) and, where present, human labels.
Scale note: our segments are 9.362 um except PHerc0343P (8.64 um); sigma kept in px (48 um -> 44 um there).
usage: hp_eval.py --segments ... --models name ... --out results_hp.json
"""
import argparse
import json
import os

import numpy as np
import tifffile
import zarr
from scipy.ndimage import gaussian_filter

OUT = os.environ.get('DIST9_DATA', 'dist9_data')
SIG, SMO = 20 / 3.9, 40 / 3.9


def hp(img, m):
    w = gaussian_filter(m.astype(np.float32), SIG)
    return img - gaussian_filter(img * m, SIG) / np.maximum(w, 1e-3)


def r(a, b):
    if a.size < 1000:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def score(key, pred, valid):
    key = np.where(valid, key, 0).astype(np.float32)
    kh = hp(key, valid)
    inked = gaussian_filter(key, SMO) > 60 / 255
    core = valid & (gaussian_filter(valid.astype(np.float32), 2 * SIG) > 0.999)
    m = core & (pred > 0)
    ph = hp(pred, m)
    res = {'hp_r': r(ph[m], kh[m]), 'inked_hp_r': r(ph[m & inked], kh[m & inked]), 'raw_r': r(pred[m], key[m]),
           'px': int(m.sum())}
    nulls = []
    for s in (150, 300, -300):
        mm = m & np.roll(m, s, 0)
        nulls.append(abs(r(ph[mm], np.roll(kh, s, 0)[mm])))
    res['null_max'] = float(np.nanmax(nulls))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', nargs='+', required=True)
    ap.add_argument('--models', nargs='+', required=True, help='names of pred_<name>.tif in each segment dir')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = {}
    for seg in a.segments:
        d = os.path.join(OUT, seg)
        sv = np.asarray(zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')[14])
        canon = np.load(os.path.join(d, 'canon_al.npy')).astype(np.float32)
        hpath = os.path.join(d, 'human_al.npy')
        human = np.load(hpath).astype(np.float32) if os.path.exists(hpath) else None
        for mname in a.models:
            p = os.path.join(d, f'pred_{mname}.tif')
            if not os.path.exists(p):
                continue
            pred = tifffile.imread(p).astype(np.float32)
            pred = pred / 255 if pred.max() > 1.5 else pred
            H = min(pred.shape[0], canon.shape[0], sv.shape[0]); W = min(pred.shape[1], canon.shape[1], sv.shape[1])
            pr, ca, v = pred[:H, :W], canon[:H, :W], (sv[:H, :W] > 0) & np.isfinite(canon[:H, :W])
            out = {'canon': score(np.nan_to_num(ca), pr, v)}
            if human is not None:
                out['human'] = score(human[:H, :W], pr, sv[:H, :W] > 0)
            res[f'{mname}|{seg}'] = out
            c = out['canon']
            print(f"{seg[-28:]:28s} {mname:16s} canon hp r {c['hp_r']:+.3f} inked {c['inked_hp_r']:+.3f} "
                  f"null {c['null_max']:.3f}" + (f" | human hp r {out['human']['hp_r']:+.3f} null {out['human']['null_max']:.3f}"
                                                  if 'human' in out else ''), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
