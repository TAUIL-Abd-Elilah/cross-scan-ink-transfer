"""Grid of ink maps over all depth windows for one crop (flat, flatm30, flatp30 variants) + CT."""
import sys, glob, re
import numpy as np
from PIL import Image, ImageDraw
crop, out = sys.argv[1], sys.argv[2]
ds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
def down(x):
    h, w = x.shape[0] // ds * ds, x.shape[1] // ds * ds
    return x[:h, :w].reshape(h // ds, ds, w // ds, ds).mean((1, 3))
items = []
for p in glob.glob(crop + '_flat*_ink_o*.npy'):
    m = re.search(r'_flat(m30|p30|m60|p60)?_ink_o([+-]\d+)\.npy$', p)
    base = {'m30': -30, 'p30': 30, 'm60': -60, 'p60': 60, None: 0}[m.group(1)]
    items.append((base + int(m.group(2)), p))
items.sort()
ct = np.load(crop + '_flat.npy', mmap_mode='r')
c = down(np.asarray(ct[ct.shape[0] // 2], np.float32)); lo, hi = np.percentile(c, [1, 99])
panels = [('CT flattened centre', np.clip((c - lo) / (hi - lo), 0, 1))]
maps = [down(np.load(p)) for _, p in items]
panels += [(f'depth {k:+d}', m) for (k, _), m in zip(items, maps)]
panels.append(('max over depths', np.max(maps, 0)))
h, w = panels[0][1].shape
cols = 2
rows = (len(panels) + cols - 1) // cols
cv = Image.new('L', (cols * (w + 6), rows * (h + 16)), 255); d = ImageDraw.Draw(cv)
for i, (n, im) in enumerate(panels):
    x, y = (i % cols) * (w + 6), (i // cols) * (h + 16)
    d.text((x + 3, y + 2), n, fill=0)
    cv.paste(Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)), (x, y + 16))
cv.save(out); print(out, cv.size, [k for k, _ in items])
