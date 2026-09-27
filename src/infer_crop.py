"""Run the canonical 2.4 um ink model (scrollprize/ink_canonical_2um) on a rendered crop.

Mirrors villa ink-detection/optimized_inference for MODEL_TYPE=resnet3d-152-3d-decoder:
layers [START, END) of the surface volume (default 1..63 = 62 layers), clip 0..200, /200,
256 tiles, sigmoid, bilinear x4 upsample of the logits map, Hann-weighted overlap-add, and
the per-tile validity mask (a pixel counts only where some layer is non-zero).

Input: <crop>.npy uint8 (layers, H, W) from render_crop.py. Output: <crop>_ink.npy float32.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

VILLA_INF = os.environ.get('VILLA_INFERENCE_DIR', 'villa/ink-detection/optimized_inference')
sys.path.insert(0, VILLA_INF)
from model_resnet3d_3d_decoder import load_model  # noqa: E402

CKPT = os.environ.get('CANON_CKPT', 'models/r152_3ddec_v2_l5_epoch13.ckpt')  # scrollprize/ink_canonical_2um


def hann2d(h, w):
    k = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return k / k.sum()


def grid_1d(L, tile, stride):
    xs = list(range(0, max(1, L - tile + 1), stride))
    end = max(0, L - tile)
    if not xs or xs[-1] != end:
        xs.append(end)
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('crop')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=63)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--stride', type=int, default=128)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--offsets', type=int, nargs='*', default=None,
                    help='run 62-layer windows centred at (centre layer + offset) of a deeper render; '
                         'writes <crop>_ink_o<offset>.npy for each')
    a = ap.parse_args()

    full = np.load(a.crop + '.npy', mmap_mode='r')
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    n = a.end - a.start
    model = load_model(CKPT, dev, num_frames=n)
    model.eval()
    if a.offsets is None:
        jobs = [(a.start, a.out or (a.crop + ('_ink_rev' if a.reverse else '_ink') + '.npy'))]
    else:
        c = (full.shape[0] - 1) // 2
        jobs = [(c - 31 + o, a.crop + f'_ink_o{o:+d}.npy') for o in a.offsets]
    for s0, path in jobs:
        vol = full[s0:s0 + n]
        if a.reverse:
            vol = vol[::-1]
        run(model, vol, a, dev, path)


def run(model, vol, a, dev, path):
    C, H, W = vol.shape
    wt = hann2d(a.tile, a.tile)
    pred = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    xy = [(x, y) for y in grid_1d(H, a.tile, a.stride) for x in grid_1d(W, a.tile, a.stride)]
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(xy), a.batch):
            b = xy[i:i + a.batch]
            tiles = np.stack([np.asarray(vol[:, y:y + a.tile, x:x + a.tile]) for x, y in b])
            valid = (tiles != 0).any(1).astype(np.float32)
            t = torch.from_numpy(np.clip(tiles, 0, 200).astype(np.float32) / 200.0)[:, None].to(dev)
            with torch.autocast('cuda'):
                y_ = model.forward(t)
            y_ = torch.sigmoid(y_.float())
            y_ = F.interpolate(y_, size=(a.tile, a.tile), mode='bilinear', align_corners=False)
            y_ = y_[:, 0].cpu().numpy()
            for (x, y), p, v in zip(b, y_, valid):
                pred[y:y + a.tile, x:x + a.tile] += p * wt * v
                cnt[y:y + a.tile, x:x + a.tile] += wt * v
            if (i // a.batch) % 25 == 0:
                print(f'  {i + len(b)}/{len(xy)} tiles {time.time() - t0:.0f}s', flush=True)
    out = np.where(cnt > 0, pred / np.maximum(cnt, 1e-12), 0).astype(np.float32)
    np.save(path, out)
    print('wrote', path, 'mean', float(out.mean()), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
