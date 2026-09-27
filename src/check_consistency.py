"""Do the per-height fine registrations of one scroll agree? Map fixed (9.36 um) points
through each transform and report pairwise displacement, in 9.36 um voxels."""
import itertools
import json
import sys

import numpy as np
import SimpleITK as sitk


def tf(d):
    t = sitk.Similarity3DTransform()
    t.SetCenter(d['center_xyz_mm'])
    t.SetParameters(list(d['versor_xyz']) + list(d['translation_xyz_mm']) + [d['scale']])
    return t


name = sys.argv[1]
zs = sys.argv[2:]
D = {z: json.load(open(f'{name}_fine_z{z}.json')) for z in zs}
for z, d in D.items():
    print(f'z{z}: ncc {d["ncc_initial"]:.3f}->{d["ncc_final"]:.3f} centre_z {d["center_xyz_mm"][2]:.2f} mm')
T = {z: tf(d) for z, d in D.items()}
# test points: a cylinder shell at radius 10/20/30 mm around each block centre, at each block's height
pts = []
for d in D.values():
    cx, cy, cz = d['center_xyz_mm']
    for r in (5, 15, 25):
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            pts.append((cx + r * np.cos(th), cy + r * np.sin(th), cz))
pts = np.array(pts)
vox = 9.362e-3
for a, b in itertools.combinations(zs, 2):
    pa = np.array([T[a].TransformPoint(tuple(p)) for p in pts])
    pb = np.array([T[b].TransformPoint(tuple(p)) for p in pts])
    dd = np.linalg.norm(pa - pb, axis=1) / vox
    print(f'z{a} vs z{b}: disagreement in 9.36um voxels median {np.median(dd):.2f} p95 {np.percentile(dd, 95):.2f} max {dd.max():.2f}')
