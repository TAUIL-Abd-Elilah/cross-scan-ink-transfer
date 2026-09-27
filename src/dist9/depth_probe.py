"""Does the 17-layer window position matter? Run a checkpoint with windows starting at 0..11 and score."""
import json, os, sys
from align_pair import OUT, infer
from eval9 import score
ckpt_name, ckpt = sys.argv[1].split('=', 1)
res = {}
for seg in sys.argv[2:]:
    for s0 in (1, 3, 6, 9, 11):
        pred = infer(os.path.join(OUT, seg), ckpt, f'depth_{ckpt_name}_s{s0}', ('--layer-start', str(s0), '--layer-end', str(s0 + 17)))
        r = score(seg, pred); res[f'{seg}|{s0}'] = r
        print(seg[-24:], 'start', s0, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
json.dump(res, open(f'depth_probe_{ckpt_name}.json', 'w'), indent=1)
