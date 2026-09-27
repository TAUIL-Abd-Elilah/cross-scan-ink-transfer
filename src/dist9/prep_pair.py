"""Build one (native 9.362 um input, 2.4 um-derived label) training pair for a PHerc0139 segment.

Outputs in $DIST9_DATA/<seg>/:
  sv.zarr      native 9.362 um surface volume (28, H, W) uint8, zarr v2, chunks (28,128,128), local copy
  canon.npy    canonical 2.399 um prediction (new_canon_autoresearch_recipe) area-downsampled by
               9.362/2.399 onto the 9.362 grid, float16 in [0, 1] (NaN outside the prediction)
  human.npy    human ink labels (20260918) on the same grid, fraction of labelled-ink 2.4 px (if any)
  meta.json    shapes, scale, sources
The grid offset between the two renders is NOT assumed here; align_pair.py measures it.
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tifffile
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from render_tifxyz import ChunkStore  # noqa: E402

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
OUT = os.environ.get('DIST9_DATA', 'dist9_data')
SCALE = 9.362 / 2.399


def get(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def main(seg):
    import re
    scroll, seg = seg.split('/', 1) if '/' in seg else ('PHerc0139', seg)
    d = os.path.join(OUT, seg if scroll == 'PHerc0139' else f'{scroll}__{seg}')
    os.makedirs(d, exist_ok=True)
    base = f'{B}/{scroll}/segments/{seg}'
    # 1. native ~9 um surface volume (8.64 / 9.36x um) -> local zarr v2
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/{seg}/surface-volumes/&delimiter=/').read().decode()
    svs = [x for x in re.findall(r'<Prefix>[^<]*/surface-volumes/([^<]*)/</Prefix>', lst) if re.match(r'(8\.6|9\.3)', x)]
    assert len(svs) == 1, svs
    zp = os.path.join(d, 'sv.zarr')
    if not os.path.exists(os.path.join(zp, 'done')):
        st = ChunkStore(f'{base}/surface-volumes/{svs[0]}', max_cached=4, workers=16)
        L, H, W = st.shape
        z = zarr.open_array(zp, mode='w', shape=(L, H, W), chunks=st.chunks, dtype='u1', zarr_format=2,
                            compressor=None, fill_value=0)
        keys = [(0, a, b) for a in range((H + 127) // 128) for b in range((W + 127) // 128)]

        def fetch(k):
            return k, st._load(k)[0]

        with ThreadPoolExecutor(16) as ex:
            for (c0, a, b), blk in ex.map(fetch, keys):
                y1, x1 = min(H, (a + 1) * 128), min(W, (b + 1) * 128)
                z[:, a * 128:y1, b * 128:x1] = blk[:, :y1 - a * 128, :x1 - b * 128]
        open(os.path.join(zp, 'done'), 'w').write('ok')
    z = zarr.open_array(zp, mode='r')
    L, H, W = z.shape
    # 2. canonical 2.399 prediction -> 9.362 grid
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/{seg}/ink-detection/').read().decode()
    keys = [k for k in re.findall(r'<Key>([^<]*)</Key>', lst)
            if re.search(r'-2\.[0-9]+um-', k) and k.endswith('.tif') and '/downsampled/' not in k]
    keys = sorted(keys, key=lambda k: 'new_canon' not in k)
    assert keys, 'no ~2.4 um prediction'
    tif = get(f'{B}/{keys[0]}', os.path.join(d, 'canon_2p4.tif'))
    p = tifffile.imread(tif)
    if p.ndim == 3:
        p = p[..., 0] if p.shape[-1] in (1, 3, 4) else p[0]
    assert p.dtype == np.uint8, p.dtype
    h2, w2 = int(round(p.shape[0] / SCALE)), int(round(p.shape[1] / SCALE))
    small = cv2.resize(p, (w2, h2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    canon = np.full((H, W), np.nan, np.float32)
    hh, ww = min(H, h2), min(W, w2)
    canon[:hh, :ww] = small[:hh, :ww]
    np.save(os.path.join(d, 'canon.npy'), canon.astype(np.float16))
    del p
    # 3. human labels (zarr v3, level 0), if published
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/{seg}/ink-labels/').read().decode()
    lz = sorted(set(re.findall(r'<Key>([^<]*inklabels\.zarr)/0/zarr\.json</Key>', lst)))
    human = None
    if lz:
        root = os.path.join(d, 'inklabels.zarr')
        os.makedirs(os.path.join(root, '0', 'c', '0'), exist_ok=True)
        get(f'{B}/{lz[-1]}/zarr.json', os.path.join(root, 'zarr.json'))
        get(f'{B}/{lz[-1]}/0/zarr.json', os.path.join(root, '0', 'zarr.json'))
        get(f'{B}/{lz[-1]}/0/c/0/0', os.path.join(root, '0', 'c', '0', '0'))
        lab = ((zarr.open_array(os.path.join(root, '0'), mode='r')[:] > 0) * 255).astype(np.uint8)
        s = cv2.resize(lab, (int(round(lab.shape[1] / SCALE)), int(round(lab.shape[0] / SCALE))), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        human = np.zeros((H, W), np.float32)
        hh, ww = min(H, s.shape[0]), min(W, s.shape[1])
        human[:hh, :ww] = s[:hh, :ww]
        np.save(os.path.join(d, 'human.npy'), human.astype(np.float16))
    json.dump({'scroll': scroll, 'sv_source': svs[0], 'segment': seg, 'sv_shape': [L, H, W], 'canon_2p4_shape': list(small.shape), 'scale': SCALE,
               'canon_source': keys[0], 'human_source': lz[-1] if lz else None,
               'human_ink_frac': float(human.mean()) if human is not None else None},
              open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    print(seg, 'sv', (L, H, W), 'canon', small.shape, 'human', None if human is None else round(float(human.mean()), 4), flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        main(s)
