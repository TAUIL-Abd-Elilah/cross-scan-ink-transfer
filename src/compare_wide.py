"""NCC template search of an ink map (ds8) over the whole published w043 prediction.
Reports best r, its location, and the distribution of r over all positions (null)."""
import json
import os, sys
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
Image.MAX_IMAGE_PIXELS = None
ink = np.load(sys.argv[1]); ds = 8
H, W = ink.shape[0] // ds * ds, ink.shape[1] // ds * ds
t = ink[:H, :W].reshape(H // ds, ds, W // ds, ds).mean((1, 3))
m = 8  # drop the border (tile edge effects)
t = t[m:-m, m:-m]
pub = np.asarray(Image.open(os.path.join(os.environ.get('XSCAN_DATA', 'data'), 'pred_2399_ds8.jpg')).convert('L'), np.float32) / 255
tz = (t - t.mean()) / (t.std() + 1e-9)
n = tz.size
k = np.ones_like(tz)
s1 = fftconvolve(pub, k[::-1, ::-1], mode='valid')
s2 = fftconvolve(pub ** 2, k[::-1, ::-1], mode='valid')
num = fftconvolve(pub, tz[::-1, ::-1], mode='valid')
var = s2 - s1 ** 2 / n
ok = var > 0.25 * pub.var() * n          # window must carry real structure
r = np.where(ok, num / np.sqrt(np.maximum(var, 1e-9) * n), -1.0)
iy, ix = np.unravel_index(np.argmax(r), r.shape)
out = {'best_r': float(r[iy, ix]), 'at_ds8_yx': [int(iy - m), int(ix - m)],
       'r_p99': float(np.percentile(r[ok], 99)), 'r_p999': float(np.percentile(r[ok], 99.9)), 'n_positions': int(ok.sum())}
if len(sys.argv) > 3:                   # r near an expected location (ds8 y x), +-R
    ey, ex, R = int(sys.argv[2]) + 0, int(sys.argv[3]), 80
    sub = r[max(0, ey - m - R):ey - m + R, max(0, ex - m - R):ex - m + R]
    jy, jx = np.unravel_index(np.argmax(sub), sub.shape)
    out['near_expected_r'] = float(sub[jy, jx])
    out['near_expected_at'] = [int(max(0, ey - m - R) + jy), int(max(0, ex - m - R) + jx)]
print(json.dumps(out))
p = pub[iy:iy + t.shape[0], ix:ix + t.shape[1]]
img = np.concatenate([np.clip(t, 0, 1), np.ones((4, t.shape[1])), p], 0)
Image.fromarray((img * 255).astype(np.uint8)).resize((t.shape[1] * 2, img.shape[0] * 2)).save(sys.argv[1][:-4] + '_wide.png')
