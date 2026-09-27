"""Height, rotation and mirror between two scans of one scroll from slice outlines.

For each slice: fill the material mask, take the largest component, measure the
boundary radius r(theta) around its centroid (1 deg rays). Rotation-invariant
descriptor: |FFT| of r/mean(r), harmonics 1..24. Each 2.4 um slice is matched to
every 9.36 um slice; the z offset is the mode of (z_b - z_a) over many slices, with
a robust line fit giving scale. Rotation/mirror: circular cross-correlation of r(theta).
"""
import json
import sys

import numpy as np
from scipy import ndimage

UM24, UM93 = 2.403, 9.362
N_THETA = 360


def outline(sl):
    m = ndimage.binary_fill_holes(sl > 0)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    if m.sum() < 200:
        return None
    cy, cx = ndimage.center_of_mass(m)
    th = np.deg2rad(np.arange(N_THETA))
    rmax = int(np.hypot(*m.shape))
    rs = np.arange(0, rmax, 0.5)
    ys = cy + np.outer(np.sin(th), rs)
    xs = cx + np.outer(np.cos(th), rs)
    inside = ndimage.map_coordinates(m.astype(np.float32), [ys.ravel(), xs.ravel()], order=0,
                                     mode='constant').reshape(ys.shape) > 0.5
    r = np.array([rs[np.flatnonzero(row)[-1]] if row.any() else 0.0 for row in inside])
    return r, (cy, cx), m.sum()


def descriptor(r):
    rn = r / (r.mean() + 1e-9)
    f = np.abs(np.fft.rfft(rn - rn.mean()))[1:25]
    return f / (np.linalg.norm(f) + 1e-9)


def main(name, f24, lvl24, f93, lvl93, step_mm=0.3):
    a = np.load(f24)
    b = np.load(f93)
    va = UM24 * 2 ** lvl24 / 1000.0
    vb = UM93 * 2 ** lvl93 / 1000.0
    za = np.arange(0, len(a) * va, step_mm)
    zb = np.arange(0, len(b) * vb, step_mm)
    oa, ob = [], []
    for z in za:
        o = outline(a[min(len(a) - 1, int(round(z / va)))])
        oa.append(o)
    for z in zb:
        o = outline(b[min(len(b) - 1, int(round(z / vb)))])
        ob.append(o)
    ia = [i for i, o in enumerate(oa) if o is not None]
    ib = [i for i, o in enumerate(ob) if o is not None]
    Da = np.stack([descriptor(oa[i][0]) for i in ia])
    Db = np.stack([descriptor(ob[i][0]) for i in ib])
    # also compare physical mean radius (scale-true) to avoid matching different sizes
    Ra = np.array([oa[i][0].mean() * va for i in ia])
    Rb = np.array([ob[i][0].mean() * vb for i in ib])
    cost = ((Da[:, None, :] - Db[None, :, :]) ** 2).sum(-1) + 4.0 * ((Ra[:, None] - Rb[None, :]) / Rb.mean()) ** 2
    zA = za[ia]
    zB = zb[ib]
    # offset search: for each candidate offset, mean cost along the matched diagonal
    offsets = np.arange(-zA[0], zB[-1] - zA[-1] + step_mm, step_mm)
    scores = []
    for off in offsets:
        j = np.searchsorted(zB, zA + off)
        ok = (j > 0) & (j < len(zB))
        if ok.sum() < 0.8 * len(zA):
            scores.append(np.inf)
            continue
        scores.append(cost[np.flatnonzero(ok), j[ok]].mean())
    scores = np.asarray(scores)
    k = int(np.argmin(scores))
    off = float(offsets[k])
    finite = np.sort(scores[np.isfinite(scores)])
    # rotation / mirror at matched pairs
    rots = {False: [], True: []}
    for i_local, i in enumerate(ia):
        jb = np.searchsorted(zB, zA[i_local] + off)
        if not 0 < jb < len(zB):
            continue
        ra = oa[i][0] / oa[i][0].mean()
        rb = ob[ib[jb]][0] / ob[ib[jb]][0].mean()
        for mirror in (False, True):
            rr = ra[::-1] if mirror else ra
            cc = np.real(np.fft.ifft(np.fft.fft(rb - rb.mean()) * np.conj(np.fft.fft(rr - rr.mean()))))
            s = int(np.argmax(cc))
            rots[mirror].append((float(cc[s] / (np.linalg.norm(rb - rb.mean()) * np.linalg.norm(rr - rr.mean()) + 1e-9)), s))
    summ = {}
    for mirror, v in rots.items():
        v = np.asarray(v)
        angs = v[:, 1]
        # circular median of angles
        c = np.angle(np.mean(np.exp(1j * np.deg2rad(angs))), deg=True) % 360
        spread = np.degrees(np.abs(np.angle(np.exp(1j * np.deg2rad(angs - c))))).mean()
        summ[mirror] = {'mean_ncc': float(v[:, 0].mean()), 'angle_deg': float(c), 'angle_spread_deg': float(spread)}
    mirror = max(summ, key=lambda m: summ[m]['mean_ncc'])
    out = {'name': name, 'step_mm': step_mm, 'z_offset_mm': off, 'offset_cost': float(scores[k]),
           'offset_cost_2nd_best_sep': float(finite[1] - finite[0]) if len(finite) > 1 else None,
           'offset_cost_median': float(np.median(finite)),
           'a_height_mm': float(zA[-1] - zA[0]), 'b_height_mm': float(zB[-1] - zB[0]),
           'mean_radius_mm_a': float(Ra.mean()), 'mean_radius_mm_b_matched': None,
           'mirror': bool(mirror), 'rotation': summ[mirror], 'other_mirror': summ[not mirror]}
    # radius sanity at the matched offset
    jb = np.searchsorted(zB, zA + off)
    ok = (jb > 0) & (jb < len(zB))
    out['mean_radius_mm_b_matched'] = float(Rb[jb[ok]].mean())
    np.save(f'{name}_offset_scores.npy', np.stack([offsets, scores]))
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{name}_outline.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
