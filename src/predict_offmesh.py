"""One command: depth-corrected canonical ink prediction for a segment on a 2.4 um scan whose mesh may be off-sheet.

  python src/predict_offmesh.py <segment.tifxyz dir> <volume zarr URL> <out prefix> [--rows r0 r1 --cols c0 c1]

What it does (validated on 8 human-labelled PHerc0139 segments on the July 2025 ROI scan 20260319133554, README section C2):
  1. renders 161 layers (about +-190 um) along the mesh's smoothed normals, reading 4.8 um data onto the 2.4 um
     grid (level 1, oversample 2): about 1.2 GB of CT per cm2;
  2. flattens the render onto the local papyrus layering (cross-correlation dip, refine_surface.py);
  3. runs scrollprize/ink_canonical_2um at 5 depth windows (-96, -48, 0, +48, +96 um);
  4. keeps the window whose map has the most pixels above 0.5 (no labels used) and writes
     <out>_ink.tif (uint8, 0-255, on the segment's 20x-upsampled grid) and <out>_depth.json (chosen depth, per-window
     confidence), plus <out>_ink_mesh_as_is.tif for comparison.
On the benchmark this lifts mean AUC against human labels from 0.756 (mesh as published) to 0.803, within 0.007 of
choosing the depth with the labels; where the mesh was already right it costs at most 0.036.
Needs CANON_CKPT and VILLA_INFERENCE_DIR (see runs/env.sh). Omit --rows/--cols to process the whole segment.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OFFS = [-40, -20, 0, 20, 40]            # render layers at 2.4 um: -96 .. +96 um


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tifxyz'); ap.add_argument('volume'); ap.add_argument('out')
    ap.add_argument('--rows', type=int, nargs=2); ap.add_argument('--cols', type=int, nargs=2)
    a = ap.parse_args()
    if not (a.rows and a.cols):
        x = tifffile.imread(os.path.join(a.tifxyz, 'x.tif'))
        a.rows, a.cols = a.rows or [0, x.shape[0] - 1], a.cols or [0, x.shape[1] - 1]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    run = lambda *c: subprocess.check_call([PY, *c])
    if not os.path.exists(a.out + '.npy'):
        run(os.path.join(HERE, 'render_crop.py'), a.tifxyz, a.volume, a.out, '--rows', *map(str, a.rows),
            '--cols', *map(str, a.cols), '--layers', '161', '--level', '1', '--oversample', '2', '--smooth-normals', '1.0')
    run(os.path.join(HERE, 'infer_crop.py'), a.out, '--offsets', '0')
    run(os.path.join(HERE, 'refine_surface.py'), a.out, '--out-layers', '145')
    run(os.path.join(HERE, 'infer_crop.py'), a.out + '_flat', '--offsets', *map(str, OFFS))
    conf = {}
    for o in OFFS:
        m = np.load(a.out + f'_flat_ink_o{o:+d}.npy')
        conf[o] = float((m > 0.5).mean())
    best = max(conf, key=conf.get)
    ink = np.load(a.out + f'_flat_ink_o{best:+d}.npy')
    tifffile.imwrite(a.out + '_ink.tif', np.clip(ink * 255, 0, 255).astype(np.uint8))
    tifffile.imwrite(a.out + '_ink_mesh_as_is.tif',
                     np.clip(np.load(a.out + '_ink_o+0.npy') * 255, 0, 255).astype(np.uint8))
    json.dump({'chosen_depth_um': best * 2.4, 'confidence_by_depth_um': {str(o * 2.4): c for o, c in conf.items()},
               'rows': a.rows, 'cols': a.cols, 'volume': a.volume, 'tifxyz': os.path.abspath(a.tifxyz)},
              open(a.out + '_depth.json', 'w'), indent=1)
    print(f'chosen depth {best * 2.4:+.0f} um -> {a.out}_ink.tif', flush=True)


if __name__ == '__main__':
    main()
