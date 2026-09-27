#!/usr/bin/env bash
# D. Resolution. Synthetic: line-1 published 2.399 sv, 2x2x2 / 4x4x4 mean-pooled then trilinear back
#    to the 2.4 grid: r 0.885 / 0.66.  Real: the published 9.362 um sv of the same segment, upsampled
#    x3.9025 to the 2.4 grid: best r near the expected spot 0.32 (p99.9 of all positions 0.27), not legible.
source runs/env.sh; mkdir -p out
python - <<'PY'
import numpy as np, json, shutil
from scipy import ndimage
v = np.load('out/l1_pubsv.npy').astype(np.float32); L, H, W = v.shape
for f in (2, 4):
    d = v[:L//f*f, :H//f*f, :W//f*f].reshape(L//f, f, H//f, f, W//f, f).mean((1, 3, 5))
    u = ndimage.zoom(d, (L/d.shape[0], H/d.shape[1], W/d.shape[2]), order=1)[:L, :H, :W]
    np.save(f'out/l1_pubsv_ds{f}.npy', np.clip(np.rint(u), 0, 255).astype(np.uint8)); shutil.copy('out/l1_pubsv.json', f'out/l1_pubsv_ds{f}.json')
PY
for f in 2 4; do python src/infer_crop.py out/l1_pubsv_ds$f --offsets 0; python src/compare_ctrl.py out/l1_pubsv_ds$f out/l1_pubsv_ds${f}_ink_o+0.npy; done
python src/fetch_sv.py $SV93 out/l1_sv936 3700 4180 2850 4420
python - <<'PY'
import numpy as np, json
from scipy import ndimage
v = np.load('out/l1_sv936.npy').astype(np.float32); f = 9.362 / 2.399
np.save('out/l1_sv936_up.npy', np.clip(np.rint(ndimage.zoom(v, (f, f, f), order=1)), 0, 255).astype(np.uint8))
json.dump({'rows': [0, 0], 'cols': [0, 0], 'upsample': 20, 'grid': 'published'}, open('out/l1_sv936_up.json', 'w'))
PY
python src/infer_crop.py out/l1_sv936_up --offsets -23 -12 0 12 23
for o in -23 -12 +0 +12 +23; do python src/compare_wide.py out/l1_sv936_up_ink_o$o.npy 1804 1390; done
python src/compare_wide.py out/l1_pubsv_ink_o+0.npy 1830 1415
