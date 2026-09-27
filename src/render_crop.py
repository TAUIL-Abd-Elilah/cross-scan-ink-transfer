"""Render a surface volume for a sub-rectangle of a tifxyz mesh (grid rows/cols).

Reuses the validated sampler from render_tifxyz.py (trilinear, corner-aligned
upsample, cross(dv, du) normal = vc_render_tifxyz layer order). Only the requested grid
window (plus a 2-cell margin for normals) is upsampled, so a small crop of a large mesh
costs only its own CT chunks.

Output: <out>.npy uint8 (layers, H, W), H = (r1 - r0) / scale, W = (c1 - c0) / scale,
and <out>.json with provenance.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_tifxyz import ChunkStore, normals, read_tifxyz, upsample  # noqa: E402
CACHE = os.environ.get('XSCAN_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache'))


class CachedStore(ChunkStore):
    """ChunkStore whose chunks persist in a local disk cache (XSCAN_CACHE) (re-renders cost no download)."""

    def __init__(self, url, **kw):
        super().__init__(url, **kw)
        vol = url.split('amazonaws.com/', 1)[1].rstrip('/')
        self.dir = os.path.join(CACHE, vol.replace('/', '__'), 'L' + kw.get('level', '0'))
        os.makedirs(self.dir, exist_ok=True)

    def _load(self, key):
        path = os.path.join(self.dir, '%d_%d_%d.bin' % key)
        if os.path.exists(path):
            raw = open(path, 'rb').read()
            if not raw:
                return np.full(self.chunks, self.fill, np.uint8), True
            return np.frombuffer(raw, np.uint8).reshape(self.chunks), False
        arr, absent = super()._load(key)
        with open(path + '.part', 'wb') as h:
            h.write(b'' if absent else arr.tobytes())
        os.replace(path + '.part', path)
        return arr, absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tifxyz_dir')
    ap.add_argument('volume_url')
    ap.add_argument('out')
    ap.add_argument('--rows', type=int, nargs=2, required=True, help='grid rows [r0, r1)')
    ap.add_argument('--cols', type=int, nargs=2, required=True, help='grid cols [c0, c1)')
    ap.add_argument('--layers', type=int, default=65)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--cache-chunks', type=int, default=3000)
    ap.add_argument('--level', type=int, default=0,
                    help='render from pyramid level L: mesh L0 coords -> (c - (2^L - 1)/2) / 2^L, '
                         'and one output px per level-L voxel')
    ap.add_argument('--oversample', type=int, default=1,
                    help='output px and layer step = 1/oversample of a level-L voxel (e.g. --level 1 '
                         '--oversample 2 reads 4.8 um data onto the 2.4 um grid the canonical model expects)')
    ap.add_argument('--smooth-normals', type=float, default=0.0,
                    help='Gaussian sigma in grid cells for the normal field (0 = off). Coarse meshes '
                         '(e.g. 9.36 um cells upsampled 78x) have piecewise-constant tangents, so the '
                         'raw normal jumps at every cell edge and far layers tear.')
    a = ap.parse_args()

    x, y, z, valid, meta = read_tifxyz(a.tifxyz_dir)
    s = float(meta['scale'][0])
    FL = 2 ** a.level
    if a.level:
        x, y, z = [np.where(valid, (c - (FL - 1) / 2.0) / FL, -1.0).astype(np.float32) for c in (x, y, z)]
    f = int(round(1.0 / s / FL * a.oversample))
    (r0, r1), (c0, c1) = a.rows, a.cols
    m = 2 + int(np.ceil(3 * a.smooth_normals))
    R0, C0 = max(0, r0 - m), max(0, c0 - m)
    R1, C1 = min(x.shape[0], r1 + m), min(x.shape[1], c1 + m)
    sub = (slice(R0, R1), slice(C0, C1))
    X, V = upsample(x[sub], valid[sub], f, f)
    Y, _ = upsample(y[sub], valid[sub], f, f)
    Z, _ = upsample(z[sub], valid[sub], f, f)
    N = normals(X, Y, Z, V)
    if a.smooth_normals > 0:
        from scipy import ndimage
        sg = a.smooth_normals * f
        N = np.stack([ndimage.gaussian_filter(N[..., i], sg, mode='nearest') for i in range(3)], -1)
        N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-8)
    oy, ox = (r0 - R0) * f, (c0 - C0) * f
    H, W = (r1 - r0) * f, (c1 - c0) * f
    crop = (slice(oy, oy + H), slice(ox, ox + W))
    X, Y, Z, V, N = X[crop], Y[crop], Z[crop], V[crop], N[crop]
    print(f'crop grid rows {r0}:{r1} cols {c0}:{c1} -> {H}x{W} px, valid {V.mean():.3f}', flush=True)

    store = CachedStore(a.volume_url, level=str(a.level), max_cached=a.cache_chunks, workers=16)
    zmax = store.shape[0] - 1
    V = V & (Z >= 0) & (Z <= zmax)
    L = a.layers
    ks = (np.arange(L, dtype=np.float32) - (L - 1) / 2.0) / a.oversample
    out = np.zeros((L, H, W), np.uint8)
    t0 = time.time()
    for y0 in range(0, H, a.tile):
        y1 = min(H, y0 + a.tile)
        for x0 in range(0, W, a.tile):
            x1 = min(W, x0 + a.tile)
            vv = V[y0:y1, x0:x1]
            if not vv.any():
                continue
            px, py, pz = X[y0:y1, x0:x1], Y[y0:y1, x0:x1], Z[y0:y1, x0:x1]
            nn = N[y0:y1, x0:x1]
            for li, k in enumerate(ks):
                smp = store.sample(pz + nn[..., 2] * k, py + nn[..., 1] * k, px + nn[..., 0] * k)
                out[li, y0:y1, x0:x1] = np.where(vv, smp, 0)
        print(f'  rows {y1}/{H} {time.time() - t0:.0f}s fetched={store.misses} absent={store.absent}',
              flush=True)
    np.save(a.out + '.npy', out)
    json.dump({'tifxyz': os.path.abspath(a.tifxyz_dir), 'volume': a.volume_url,
               'rows': [r0, r1], 'cols': [c0, c1], 'upsample': f, 'layers': L, 'level': a.level, 'oversample': a.oversample,
               'layer_offsets_vox': [float(ks[0]), float(ks[-1])], 'smooth_normals': a.smooth_normals,
               'chunks_fetched': store.misses, 'seconds': time.time() - t0},
              open(a.out + '.json', 'w'), indent=1)
    print('wrote', a.out + '.npy', flush=True)


if __name__ == '__main__':
    main()
