#!/usr/bin/env bash
# E. PHerc1203 negative screen on the March-2026 2.4 um scan (~5.8 cm2 of clean sheet surface).
# 1) register the 2.4 um scan to the eligible 9.362 um scan (outline descriptor, then Similarity3D
#    on 38 um blocks at two heights; results/1203_*.json), 2) map four published auto_grown segments
#    into the 2.4 um frame, 3) survey each whole segment at 19 um (level 3) to find clean sheet
#    surface, 4) screen clean regions in tiles: 4.8 um data on the 2.4 grid, 241 layers, flatten,
#    canonical model at 15 depths, 5) score (sheet_depths.py / flag_view.py) and look at every flag.
source runs/env.sh; mkdir -p out/1203
V93=PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr
V24=$BUCKET/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr
python - <<'PY'
import numpy as np, sys; sys.path.insert(0, 'src'); import zfetch as zf
np.save('out/1203/1203_24_L5.npy', zf.read_level('PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr', 5))
np.save('out/1203/1203_93_L4.npy', zf.read_level('PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr', 4))
PY
(cd out/1203 && python ../../src/register_outline.py 1203 1203_24_L5.npy 5 1203_93_L4.npy 4 \
  && python ../../src/register_fine.py 1203 0.25 && python ../../src/register_fine.py 1203 0.5 \
  && python ../../src/check_consistency.py 1203 25 50)
for s in 20251005230830031:11 20251005231446965:6 20251005221856743:12 20250925223153537:; do
  id=${s%%:*}; v=${s##*:}; P=PHerc1203/segments/raw/auto_grown_$id; [ -n "$v" ] && P=$P/versions/$v
  mkdir -p out/1203/seg_$id; for f in meta.json x.tif y.tif z.tif; do curl -s -o out/1203/seg_$id/$f $BUCKET/$P/$f; done
  python src/map_mesh.py out/1203/seg_$id out/1203/seg_${id}_on24 --tf out/1203/1203_fine_z25.json out/1203/1203_fine_z50.json
done
# one screening tile (the others differ only in segment and grid window; see results/pherc1203_screen_results.json)
M=out/1203/seg_20251005230830031_on24; C=out/1203/s031_r90_c20
python src/render_crop.py $M $V24 $C --rows 90 116 --cols 20 72 --layers 241 --level 1 --oversample 2 --smooth-normals 1.0
python src/refine_surface.py $C --out-layers 121
python src/infer_crop.py ${C}_flat --offsets -24 -12 0 12 24
for sh in -30 30 -60 60; do
  t=flat$( [ $sh -lt 0 ] && echo m${sh#-} || echo p$sh )
  python src/refine_surface.py $C --out-layers 121 --shift $sh --tag $t --load-d ${C}_flat_d.npy
  case $sh in -30) o="-18 -6";; 30) o="6 18";; -60) o="-24 -12 0";; 60) o="0 12 24";; esac
  python src/infer_crop.py ${C}_$t --offsets $o
done
python src/sheet_depths.py $C ${C}_depths.png 6
