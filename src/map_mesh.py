"""Map a tifxyz mesh from the eligible 9.36 um scan into the 2.4 um scan of the same scroll.

Uses the fine similarity transforms from register_fine.py (fixed = 9.36 um, moving = 2.4 um).
register_fine.py placed pyramid-level voxel i at physical i * (2**level * um); with 2x2x2
mean pooling that voxel's centre is really at L0 index 2**level * i + (2**level - 1) / 2, so
L0 points are shifted by that half-block before and after the transform.
With several transforms (different heights) the one whose block centre is nearest in z is used
per point, with linear blending between neighbours.

Output tifxyz: coordinates in 2.4 um L0 voxels (x, y, z), same grid, scale = 9.36 scale * 2.403/9.362
(one grid step still spans the same physical distance).
"""
import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
import tifffile

UM24, UM93 = 2.403, 9.362


def load_tf(path):
    d = json.load(open(path))
    t = sitk.Similarity3DTransform()
    t.SetCenter(d['center_xyz_mm'])
    t.SetParameters(list(d['versor_xyz']) + list(d['translation_xyz_mm']) + [d['scale']])
    la, lb = d['levels']
    return t, d['center_xyz_mm'][2], la, lb


def apply(t, la, lb, P93):
    """P93: (n, 3) L0 voxel xyz in the 9.36 um scan -> (n, 3) L0 voxel xyz in the 2.4 um scan."""
    fb = 2 ** lb
    fa = 2 ** la
    phys = (P93 - (fb - 1) / 2.0) * UM93 / 1000.0
    out = np.array([t.TransformPoint(tuple(map(float, p))) for p in phys])
    return out * 1000.0 / UM24 + (fa - 1) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seg_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--tf', nargs='+', required=True)
    a = ap.parse_args()
    x, y, z = [tifffile.imread(os.path.join(a.seg_dir, f'{c}.tif')).astype(np.float64) for c in 'xyz']
    meta = json.load(open(os.path.join(a.seg_dir, 'meta.json')))
    v = (x > -0.5) & (y > -0.5) & (z > -0.5)
    P = np.stack([x[v], y[v], z[v]], 1)
    tfs = sorted([load_tf(p) for p in a.tf], key=lambda r: r[1])
    maps = np.stack([apply(t, la, lb, P) for t, _, la, lb in tfs])      # (k, n, 3)
    if len(tfs) == 1:
        Q = maps[0]
    else:
        zc = np.array([c for _, c, _, _ in tfs])                       # block centres, mm, 9.36 frame
        zp = P[:, 2] * UM93 / 1000.0
        w = np.zeros((len(tfs), len(zp)))
        j = np.clip(np.searchsorted(zc, zp), 1, len(zc) - 1)
        t = np.clip((zp - zc[j - 1]) / (zc[j] - zc[j - 1]), 0, 1)
        w[j - 1, np.arange(len(zp))] = 1 - t
        w[j, np.arange(len(zp))] = t
        Q = (maps * w[..., None]).sum(0)
    os.makedirs(a.out_dir, exist_ok=True)
    for i, c in enumerate('xyz'):
        arr = np.full(x.shape, -1.0, np.float32)
        arr[v] = Q[:, i]
        tifffile.imwrite(os.path.join(a.out_dir, f'{c}.tif'), arr)
    s = [float(meta['scale'][0]) * UM24 / UM93, float(meta['scale'][1]) * UM24 / UM93]
    json.dump({'format': 'tifxyz', 'type': 'seg', 'uuid': meta.get('uuid', ''), 'scale': s,
               'mapped_from': os.path.abspath(a.seg_dir), 'transforms': a.tf,
               'bbox': [Q.min(0).tolist(), Q.max(0).tolist()]},
              open(os.path.join(a.out_dir, 'meta.json'), 'w'), indent=1)
    print('mapped', int(v.sum()), 'points; bbox 2.4um vox', Q.min(0).round(0), Q.max(0).round(0))


if __name__ == '__main__':
    main()
