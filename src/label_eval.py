"""Score an ink map against the human ink labels of PHerc0139 w043 (ds8 grid of the 2.399 um
canvas). The map's expected ds8 position is given; the best alignment within +-R ds8 px is found by
NCC against the labels, then AUC (ink prob at labelled-ink vs labelled-background pixels) and
Pearson r are reported there. Null: AUC at 12 offsets 40-80 ds8 px away (other text/background)."""
import json, os, sys
import numpy as np
ink_path, ey, ex = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
R = int(sys.argv[4]) if len(sys.argv) > 4 else 24
lab = np.load(os.path.join(os.environ.get('XSCAN_DATA', 'data'), 'labels_ds8.npy'))
ink = np.load(ink_path); ds = 8
H, W = ink.shape[0] // ds * ds, ink.shape[1] // ds * ds
t = ink[:H, :W].reshape(H // ds, ds, W // ds, ds).mean((1, 3))
h, w = t.shape


def auc(score, y):
    pos, neg = score[y], score[~y]
    if pos.size < 20 or neg.size < 20:
        return np.nan
    allv = np.concatenate([pos, neg]); rank = allv.argsort().argsort() + 1
    return float((rank[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def at(dy, dx):
    y, x = ey + dy, ex + dx
    if y < 0 or x < 0 or y + h > lab.shape[0] or x + w > lab.shape[1]:
        return None
    return lab[y:y + h, x:x + w]


best = None
for dy in range(-R, R + 1, 2):
    for dx in range(-R, R + 1, 2):
        L = at(dy, dx)
        if L is None or L.std() == 0:
            continue
        a, b = t - t.mean(), L - L.mean()
        r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        if best is None or r > best[0]:
            best = (r, dy, dx)
r, dy, dx = best
L = at(dy, dx) > 0.5
out = {'r': r, 'offset': [dy, dx], 'auc': auc(t, L), 'label_frac': float(L.mean())}
nulls = []
for ny in (-80, -40, 40, 80):
    for nx in (-80, 0, 80):
        Ln = at(dy + ny, dx + nx)
        if Ln is not None and (Ln > 0.5).mean() > 0.005:
            nulls.append(auc(t, Ln > 0.5))
nulls = [v for v in nulls if np.isfinite(v)]
out['null_auc_mean'] = float(np.mean(nulls)) if nulls else None
out['null_auc_max'] = float(np.max(nulls)) if nulls else None
print(json.dumps(out))
