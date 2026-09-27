"""Per-map ink statistics: fraction > 0.5 over the map, and the max over 2 mm windows (832 px)
of that fraction, for every *_ink*.npy matching the given globs."""
import glob, os, sys
import numpy as np
from scipy import ndimage
rows = []
for g in sys.argv[1:]:
    for p in sorted(glob.glob(g)):
        a = np.load(p, mmap_mode='r')
        a = np.asarray(a[::4, ::4])
        b = (a > 0.5).astype(np.float32)
        w = 832 // 4
        if min(b.shape) > w:
            loc = ndimage.uniform_filter(b, w)[w // 2:-w // 2, w // 2:-w // 2]
            mx = float(loc.max())
        else:
            mx = float(b.mean())
        rows.append((os.path.basename(p), float(b.mean()), mx))
for n, f, m in rows:
    print(f'{n:55s} frac>0.5 {f:.3f}  max 2mm-window {m:.3f}')
