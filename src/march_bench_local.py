"""Is the mesh offset constant along a segment? (run after march_bench.py; uses its flattened depth maps)
  subwindows: the 8.2 mm window cut into 4 sub-windows of ~2 mm along its length; AUC vs labels of each of the
              5 depth windows in each sub-window (alignment as found for the whole window) -> best depth per
              sub-window (this one uses the labels; None where a depth has too few labelled pixels to score);
  local_pick: label-free pick made locally instead of once per window: at each pixel keep the depth whose ink
              map has the most pixels > 0.5 (or the highest mean) in a 4 / 2 / 1 mm box around it; AUC of the
              composite map vs labels. 'frac_whole' is the per-window pick of march_bench_labelfree.py.
Writes results/C2_march_scan_local_depth.json and prints the table."""
import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from march_bench import DS, OUT, auc, down, eval_map, labels
OFFS = (-40, -20, 0, 20, 40)                       # layers at 2.4 um
BOX = {'4mm': 1668, '2mm': 834, '1mm': 417}          # px at 2.4 um
res = {}
for seg in sys.argv[1:]:
    d = os.path.join(OUT, seg); r = json.load(open(os.path.join(d, 'result.json')))
    lab8 = down(labels(seg, d))
    e0 = (r['window_cells'][0] * 20 // DS, r['window_cells'][2] * 20 // DS)
    M = np.stack([np.load(os.path.join(d, f'march_flat_ink_o{o:+d}.npy')).astype(np.float32) for o in OFFS])
    sub = {}
    for i, o in enumerate(OFFS):
        t = down(M[i]); h, w = t.shape
        dy, dx = r['march_flat_depths'][str(o)]['offset']
        L = lab8[e0[0] + dy:e0[0] + dy + h, e0[1] + dx:e0[1] + dx + w]
        neg = (ndimage.distance_transform_edt(L < 0.5) <= 2.0 / (2.399e-3 * DS)) & (L < 0.05)
        a = [auc(t[:, k * w // 4:(k + 1) * w // 4], (L >= 0.5)[:, k * w // 4:(k + 1) * w // 4],
                 neg[:, k * w // 4:(k + 1) * w // 4]) for k in range(4)]
        sub[o * 2.4] = [None if np.isnan(x) else x for x in a]
    best = [None if any(sub[o][k] is None for o in sub) else max(sub, key=lambda o: sub[o][k]) for k in range(4)]
    loc = {'frac_whole': r['march_pick_conf']['auc']}
    for name, S in BOX.items():
        for mode in ('frac', 'mean'):
            v = (M > 0.5).astype(np.float32) if mode == 'frac' else M
            c = np.stack([ndimage.uniform_filter(down(x), S // DS, mode='reflect') for x in v])
            k = np.kron(c.argmax(0), np.ones((DS, DS), np.int64))
            k = np.pad(k, ((0, M.shape[1] - k.shape[0]), (0, M.shape[2] - k.shape[1])), mode='edge')
            loc[f'{mode}_{name}'] = eval_map(np.take_along_axis(M, k[None], 0)[0], lab8, e0)['auc']
    res[seg] = {'subwindow_auc_by_depth_um': {str(o): v for o, v in sub.items()}, 'subwindow_best_depth_um': best,
                'local_pick_auc': loc}
    print(seg.split('-')[1][:4], 'best depth per 2 mm sub-window (um):', best,
          ' local pick:', ' '.join(f'{k} {v:.3f}' for k, v in loc.items()), flush=True)
keys = list(next(iter(res.values()))['local_pick_auc'])
mean = {k: float(np.mean([v['local_pick_auc'][k] for v in res.values()])) for k in keys}
print('MEAN', ' '.join(f'{k} {v:.3f}' for k, v in mean.items()))
res['mean_local_pick_auc'] = mean
os.makedirs('../results', exist_ok=True)
json.dump(res, open('../results/C2_march_scan_local_depth.json', 'w'), indent=1)
