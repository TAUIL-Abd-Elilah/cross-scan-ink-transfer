"""Flatten a deep surface render onto the local papyrus layering (horizon flattening).

A mesh carried over from another scan (or a coarse 9.36 um mesh) is off the sheet by tens of
2.4 um voxels and undulates relative to it; the canonical ink model loses most of its signal
beyond ~10 layers of misplacement (control: r 0.93 centred, 0.81-0.52 at +-20 layers).

Input: <crop>.npy (L, H, W) rendered along (smoothed) mesh normals, centre layer = mesh.
1. In-plane downsample by q; 3D structure tensor (Gaussian derivatives, smoothed).
2. At the current surface d(y, x) read the layer normal (principal eigenvector) and turn it into
   slopes dk/dy, dk/dx, with coherence as weight.
3. Weighted least squares for d: |grad d - slope|^2 + alpha * |d - 0|^2 (alpha keeps the sheet whose
   mean depth is nearest the mesh). Repeat with slopes re-read at the new surface.
4. Resample a 65-layer stack centred on d (linear along the render normal).
Writes <crop>_flat.npy (65, H, W) and <crop>_flat_d.npy (H/q, W/q) offsets in layers.
"""
import argparse
import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr


def tensor(S, sg, st):
    g = [ndimage.gaussian_filter(S, sg, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    T = {}
    for i in range(3):
        for j in range(i, 3):
            T[i, j] = ndimage.gaussian_filter(g[i] * g[j], st)
    return T


def slopes_at(T, d, c, clip):
    L, h, w = T[0, 0].shape
    k = np.clip(np.rint(c + d).astype(int), 0, L - 1)
    yy, xx = np.mgrid[0:h, 0:w]
    M = np.zeros((h, w, 3, 3), np.float64)
    for (i, j), A in T.items():
        M[..., i, j] = A[k, yy, xx]
        M[..., j, i] = A[k, yy, xx]
    ev, evec = np.linalg.eigh(M)
    n = evec[..., :, 2]                                   # largest eigenvalue = across layers
    nk = np.where(np.abs(n[..., 0]) < 1e-3, 1e-3, n[..., 0])
    sy = np.clip(-n[..., 1] / nk, -clip, clip)
    sx = np.clip(-n[..., 2] / nk, -clip, clip)
    coh = ((ev[..., 2] - ev[..., 1]) / (ev[..., 2] + ev[..., 1] + 1e-9)) ** 2
    return sy, sx, coh


def xcorr_slopes(P, d, c, smax, sig_k, sig_xy):
    """Dip by local cross-correlation: shift (layers) between neighbouring depth profiles,
    within a Gaussian window (sig_k layers) around the current surface. P is high-passed along k."""
    L, h, w = P.shape
    k = np.arange(L)[:, None, None]
    win = np.exp(-0.5 * ((k - c - d[None]) / sig_k) ** 2).astype(np.float32)
    out = []
    for ax in (1, 2):
        a = P * win
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
        A, B = a[tuple(sl0)], P[tuple(sl1)]
        cc = []
        for s_ in range(-smax, smax + 1):
            Bs = np.roll(B, -s_, axis=0)          # B[k + s]
            num = (A * Bs).sum(0)
            cc.append(ndimage.gaussian_filter(num, sig_xy))
        cc = np.stack(cc)
        i = np.clip(np.argmax(cc, 0), 1, 2 * smax - 1)
        y0 = np.take_along_axis(cc, (i - 1)[None], 0)[0]
        y1 = np.take_along_axis(cc, i[None], 0)[0]
        y2 = np.take_along_axis(cc, (i + 1)[None], 0)[0]
        den = y0 - 2 * y1 + y2
        frac = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        shift = (i - smax) + np.clip(frac, -0.5, 0.5)
        norm = np.sqrt((A * A).sum(0) * (B * B).sum(0)) + 1e-9
        conf = np.clip(ndimage.gaussian_filter(y1, 0) / ndimage.gaussian_filter(norm, sig_xy), 0, 1)
        # pad back to (h, w): the slope between j and j+1 is assigned to j; last row/col repeats
        pad = [(0, 0), (0, 0)]
        pad[ax - 1] = (0, 1)
        out.append((np.pad(shift, pad, mode='edge'), np.pad(conf, pad, mode='edge')))
    (sy, cy), (sx, cx) = out
    return sy, sx, np.minimum(cy, cx)


def solve(sy, sx, wgt, alpha):
    h, w = sy.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    # d[y+1,x] - d[y,x] = mean slope_y
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    ww = np.sqrt(0.5 * (wgt[:-1, :] + wgt[1:, :])).ravel()
    m = len(a)
    rows += [np.arange(r, r + m)] * 2
    cols += [b, a]
    vals += [ww, -ww]
    rhs.append(ww * 0.5 * (sy[:-1, :] + sy[1:, :]).ravel())
    r += m
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    ww = np.sqrt(0.5 * (wgt[:, :-1] + wgt[:, 1:])).ravel()
    m = len(a)
    rows += [np.arange(r, r + m)] * 2
    cols += [b, a]
    vals += [ww, -ww]
    rhs.append(ww * 0.5 * (sx[:, :-1] + sx[:, 1:]).ravel())
    r += m
    rows.append(np.arange(r, r + n))
    cols.append(idx.ravel())
    vals.append(np.full(n, np.sqrt(alpha)))
    rhs.append(np.zeros(n))
    r += n
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n))
    d = lsqr(A, np.concatenate(rhs), atol=1e-6, btol=1e-6, iter_lim=3000)[0]
    return d.reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('crop')
    ap.add_argument('--q', type=int, default=4)
    ap.add_argument('--sigma-grad', type=float, default=1.5)
    ap.add_argument('--sigma-tensor', type=float, default=4.0)
    ap.add_argument('--alpha', type=float, default=1e-3)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--out-layers', type=int, default=65)
    ap.add_argument('--shift', type=float, default=0.0, help='extra constant offset (layers) after flattening')
    ap.add_argument('--method', choices=['tensor', 'xcorr'], default='xcorr')
    ap.add_argument('--sig-k', type=float, default=12.0)
    ap.add_argument('--sig-xy', type=float, default=2.0)
    ap.add_argument('--tag', default='flat')
    ap.add_argument('--load-d', default=None, help='reuse an existing *_d.npy (skip the solve)')
    a = ap.parse_args()
    t0 = time.time()
    S = np.load(a.crop + '.npy')
    L, H, W = S.shape
    c = (L - 1) / 2.0
    q = a.q
    h, w = H // q, W // q
    Sq = S[:, :h * q, :w * q].reshape(L, h, q, w, q).astype(np.float32).mean((2, 4))
    if a.load_d:
        a.iters = 0
    elif a.method == 'tensor':
        T = tensor(Sq, a.sigma_grad, a.sigma_tensor)
    else:
        P = Sq - ndimage.gaussian_filter1d(Sq, 8.0, axis=0)
        P = ndimage.gaussian_filter(P, (1.0, 0.7, 0.7))
    # slopes are in layers per downsampled px along y/x
    d = np.load(a.load_d).astype(np.float64) if a.load_d else np.zeros((h, w))
    for it in range(a.iters):
        if a.method == 'tensor':
            sy, sx, coh = slopes_at(T, d, c, clip=2.0 * q)
        else:
            sy, sx, coh = xcorr_slopes(P, d, c, smax=2 * q, sig_k=a.sig_k, sig_xy=a.sig_xy)
        d = solve(sy, sx, coh + 1e-3, a.alpha)
        d = np.clip(d, -c + 33, c - 33)
        print(f'iter {it}: d range {d.min():.1f}..{d.max():.1f} (p5 {np.percentile(d, 5):.1f}, p95 {np.percentile(d, 95):.1f}), '
              f'coherence mean {coh.mean():.2f}, {time.time() - t0:.0f}s', flush=True)
    d = d + a.shift
    np.save(a.crop + f'_{a.tag}_d.npy', d.astype(np.float32))
    D = ndimage.zoom(d, (H / h, W / w), order=1)[:H, :W]
    D = np.pad(D, ((0, H - D.shape[0]), (0, W - D.shape[1])), mode='edge')
    Lo = a.out_layers
    out = np.zeros((Lo, H, W), np.uint8)
    kk = np.arange(Lo) - (Lo - 1) // 2
    yy, xx = np.mgrid[0:H, 0:W]
    for i, k in enumerate(kk):
        z = np.clip(c + D + k, 0, L - 1.001)
        z0 = np.floor(z).astype(int)
        f = (z - z0).astype(np.float32)
        out[i] = np.rint(S[z0, yy, xx] * (1 - f) + S[z0 + 1, yy, xx] * f).astype(np.uint8)
    np.save(a.crop + f'_{a.tag}.npy', out)
    print('wrote', a.crop + f'_{a.tag}.npy', out.shape, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
