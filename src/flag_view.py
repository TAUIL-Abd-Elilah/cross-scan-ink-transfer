"""Zoom on the highest-scoring 2 mm window of a flagged ink map: ink map, the same window at the
neighbouring depths, and the flattened CT centre layer."""
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
crop, main_ink, others, out = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
a = np.load(main_ink)
b = (a[::4, ::4] > 0.5).astype(np.float32); w = 832 // 4
loc = ndimage.uniform_filter(b, w)
iy, ix = np.unravel_index(np.argmax(loc), loc.shape)
cy, cx = iy * 4, ix * 4
H = 1400
y0, x0 = max(0, cy - H // 2), max(0, cx - H // 2)
y1, x1 = min(a.shape[0], y0 + H), min(a.shape[1], x0 + H)
def tile(img, lab):
    im = Image.fromarray((np.clip(img[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)).resize(((x1 - x0) // 3, (y1 - y0) // 3))
    d = ImageDraw.Draw(im); d.text((4, 4), lab, fill=255); return im
ct = np.load(crop + '_centre.npy').astype(np.float32)
lo, hi = np.percentile(ct[ct > 0], [1, 99]); ctn = (ct - lo) / (hi - lo)
tiles = [tile(ctn, 'CT centre'), tile(a, main_ink.split('_ink_')[-1][:-4] + ' (flagged)')]
for p in others:
    tiles.append(tile(np.load(p), p.split('_flat')[-1][:-4]))
W = sum(t.size[0] for t in tiles) + 6 * len(tiles)
cv = Image.new('L', (W, tiles[0].size[1]), 128); x = 0
for t in tiles:
    cv.paste(t, (x, 0)); x += t.size[0] + 6
cv.save(out); print(out, 'window centre px', cy, cx)
