"""Fetch a (layers, H, W) box from a published segment surface-volume zarr (uncompressed u1)."""
import sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_tifxyz import ChunkStore
url, out, y0, y1, x0, x1 = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
st = ChunkStore(url, max_cached=10, workers=16)
L, ch = st.shape[0], st.chunks
keys = [(0, a, b) for a in range(y0 // ch[1], (y1 - 1) // ch[1] + 1) for b in range(x0 // ch[2], (x1 - 1) // ch[2] + 1)]
res = np.zeros((L, y1 - y0, x1 - x0), np.uint8)
def get(k):
    return k, st._load(k)[0]
with ThreadPoolExecutor(16) as ex:
    for (cz, a, b), blk in ex.map(get, keys):
        ya, xa = a * ch[1], b * ch[2]
        sy0, sy1 = max(y0, ya), min(y1, ya + ch[1]); sx0, sx1 = max(x0, xa), min(x1, xa + ch[2])
        res[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = blk[:, sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa]
np.save(out + '.npy', res)
print('wrote', out, res.shape, len(keys), 'chunks')
