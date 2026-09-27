"""Rewrite canon_al.npy for PHerc0139 segments with the fixed systematic offset measured on the
held-out segments (w030 -8,-7; w045 -7,-7; w043 -8,-8), instead of noisy per-segment estimates."""
import os, sys
import numpy as np
import zarr
from align_pair import OUT, shift, stretched
DY, DX = -8, -7
for seg in sys.argv[1:]:
    d = os.path.join(OUT, seg)
    sv = np.asarray(zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')[14])
    H, W = sv.shape
    canon, human = stretched(d, H, W)
    np.save(os.path.join(d, 'canon_fx.npy'), shift(np.where(sv > 0, canon, np.nan), DY, DX).astype(np.float16))
    print(seg, 'fixed offset written', flush=True)
