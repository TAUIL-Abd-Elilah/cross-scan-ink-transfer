"""Positive control: our 2.4 um render (March-2026 scan 20260319133554) + canonical 2 um model
vs the published canonical prediction of the same segment (scan 20260102150214).

Grids differ slightly between the two mesh versions; valid-mask matching gave
a[r + 4, c + 5] ~ b[r, c]. The residual offset is found by a +-24 px (ds8) search.
Null: the same score at offsets of 30-60 ds8 px, i.e. against other parts of the text.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
crop = sys.argv[1]
ink_path = sys.argv[2] if len(sys.argv) > 2 else crop + '_ink.npy'
meta = json.load(open(crop + '.json'))
ink = np.load(ink_path)
gdy, gdx = (0, 0) if meta.get('grid') == 'published' else (4, 5)
(r0, r1), (c0, c1) = meta['rows'], meta['cols']
f = meta['upsample']
ds = 8
H, W = ink.shape
ours = ink[:H // ds * ds, :W // ds * ds].reshape(H // ds, ds, W // ds, ds).mean((1, 3))
pub = np.asarray(Image.open(os.path.join(os.environ.get('XSCAN_DATA', 'data'), 'pred_2399_ds8.jpg')).convert('L'), np.float32) / 255
k = f / ds                                     # ds8 px per grid cell
br0, bc0 = int(round((r0 - gdy) * k)), int(round((c0 - gdx) * k))
h, w = ours.shape


def score(dy, dx):
    y, x = br0 + dy, bc0 + dx
    if y < 0 or x < 0 or y + h > pub.shape[0] or x + w > pub.shape[1]:
        return np.nan
    p = pub[y:y + h, x:x + w]
    a, b = ours - ours.mean(), p - p.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


S = {(dy, dx): score(dy, dx) for dy in range(-24, 25) for dx in range(-24, 25)}
best = max((v, kk) for kk, v in S.items() if np.isfinite(v))
null = [score(dy, dx) for dy in (-60, -45, -30, 30, 45, 60) for dx in (-60, 0, 60)]
null = [v for v in null if np.isfinite(v)]
out = {'best_r': best[0], 'best_offset_ds8': best[1], 'null_r_mean': float(np.mean(null)),
       'null_r_max': float(np.max(null)), 'ours_mean': float(ink.mean()),
       'ours_frac_gt_0.5': float((ink > 0.5).mean())}
print(json.dumps(out, indent=1))
tag = ink_path[:-4]
json.dump(out, open(tag + '_compare.json', 'w'), indent=1)
dy, dx = best[1]
p = pub[br0 + dy:br0 + dy + h, bc0 + dx:bc0 + dx + w]
img = np.concatenate([ours, np.ones((4, w)), p], 0)
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((w * 2, img.shape[0] * 2)).save(tag + '_compare.png')
