"""Label-free depth handling for the March-scan benchmark (no peeking at labels):
  maxdepth: pixel-wise max over the 5 flattened depth windows (-40..+40);
  pick_conf: the single depth whose ink map has the largest fraction of pixels > 0.5.
Adds these to each result.json and prints the table."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from march_bench import OUT, eval_map, down, labels
segs = sys.argv[1:]
rows = []
for seg in segs:
    d = os.path.join(OUT, seg); r = json.load(open(os.path.join(d, 'result.json')))
    lab8 = down(labels(seg, d))
    r0, c0 = r['window_cells'][0], r['window_cells'][2]
    e0 = (r0 * 20 // 8, c0 * 20 // 8)
    maps = {o: np.load(os.path.join(d, f'march_flat_ink_o{int(o):+d}.npy')) for o in r['march_flat_depths']}
    r['march_maxdepth'] = eval_map(np.max(list(maps.values()), 0), lab8, e0)
    conf = {o: float((m > 0.5).mean()) for o, m in maps.items()}
    oc = max(conf, key=conf.get)
    r['march_pick_conf'] = dict(r['march_flat_depths'][oc], depth=int(oc))
    json.dump(r, open(os.path.join(d, 'result.json'), 'w'), indent=1)
    rows.append((seg.split('-')[1][:4], r['reference_2399']['auc'], r['march_mesh_as_is']['auc'], r['march_maxdepth']['auc'],
                 r['march_pick_conf']['auc'], r['march_pick_conf']['depth'], r['march_best_depth']['auc'], r['march_best_depth']['depth']))
print(f"{'seg':6s} {'ref2399':>8s} {'as-is':>7s} {'maxdep':>7s} {'pickconf':>9s} {'(d)':>5s} {'oracle':>7s} {'(d)':>5s}")
for s, a, b, c, e, ed, o, od in rows:
    print(f"{s:6s} {a:8.3f} {b:7.3f} {c:7.3f} {e:9.3f} {ed:5d} {o:7.3f} {od:5d}")
A = np.array([x[1:5] + (x[6],) for x in rows], float)
print(f"{'MEAN':6s} {A[:,0].mean():8.3f} {A[:,1].mean():7.3f} {A[:,2].mean():7.3f} {A[:,3].mean():9.3f} {'':5s} {A[:,4].mean():7.3f}")
