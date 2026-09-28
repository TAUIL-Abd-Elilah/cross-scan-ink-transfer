#!/usr/bin/env bash
# B. Same line on the July 2025 2.4 um ROI scan (volume 20260319133554, exported March 2026) via the published on-scan mesh.
#    B1 mesh as published, 65 layers: r 0.11 (off-sheet).  B2 deep render + flattening + depth
#    search: r 0.76 at -46 layers.  B3 the same from 4.8 um data (level 1, oversample 2): r 0.73.
source runs/env.sh; mkdir -p out; M=$XSCAN_DATA/mesh_20260319133554
python src/render_crop.py $M $VOL_MARCH out/l1_march --rows 736 808 --cols 571 855 --layers 65
python src/infer_crop.py out/l1_march && python src/compare_ctrl.py out/l1_march
python src/render_crop.py $M $VOL_MARCH out/l1_march_deep --rows 736 808 --cols 571 855 --layers 161 --smooth-normals 1.0
python src/refine_surface.py out/l1_march_deep --out-layers 121 --shift -30 --tag flatm30
cp out/l1_march_deep.json out/l1_march_deep_flatm30.json
python src/infer_crop.py out/l1_march_deep_flatm30 --offsets -16 -8 0 8 16
for o in -16 -8 +0 +8 +16; do python src/compare_ctrl.py out/l1_march_deep_flatm30 out/l1_march_deep_flatm30_ink_o$o.npy; done
python src/render_crop.py $M $VOL_MARCH out/l1_march_L1x2 --rows 736 808 --cols 571 855 --layers 161 --level 1 --oversample 2 --smooth-normals 1.0
python src/refine_surface.py out/l1_march_L1x2 --out-layers 121 --shift -30 --tag flatm30
cp out/l1_march_L1x2.json out/l1_march_L1x2_flatm30.json
python src/infer_crop.py out/l1_march_L1x2_flatm30 --offsets -16 -8 0
for o in -16 -8 +0; do python src/compare_ctrl.py out/l1_march_L1x2_flatm30 out/l1_march_L1x2_flatm30_ink_o$o.npy; done
