#!/usr/bin/env bash
# Public reference data for PHerc0139 segment w043 (~25 MB) into $XSCAN_DATA (default ./data).
set -e
D=${XSCAN_DATA:-data}; mkdir -p $D
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
S=PHerc0139/segments/20260112000000-w043_2026011217
curl -s -o $D/pred_2399_ds8.jpg "$B/$S/ink-detection/downsampled/PHerc0139-20260112000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg"
for m in 20260319133554-2.403um 20250728140407-9.362um; do
  mkdir -p $D/mesh_${m%%-*}
  for f in meta.json x.tif y.tif z.tif; do curl -s -o $D/mesh_${m%%-*}/$f "$B/$S/mesh/20260112000000-on-$m.tifxyz/$f"; done
done
L=$S/ink-labels/2.399um-volume-20260102150214/20260918/inklabels.zarr
mkdir -p $D/inklabels.zarr/0/c/0
curl -s -o $D/inklabels.zarr/zarr.json "$B/$L/zarr.json"
curl -s -o $D/inklabels.zarr/0/zarr.json "$B/$L/0/zarr.json"
curl -s -o $D/inklabels.zarr/0/c/0/0 "$B/$L/0/c/0/0"
python - <<PY
import numpy as np, zarr
lab = zarr.open_array('$D/inklabels.zarr/0', mode='r')[:]
ds = 8; H, W = lab.shape[0] // ds * ds, lab.shape[1] // ds * ds
np.save('$D/labels_ds8.npy', (lab[:H, :W] > 0).reshape(H // ds, ds, W // ds, ds).mean((1, 3)).astype(np.float32))
PY
echo "reference data in $D"
