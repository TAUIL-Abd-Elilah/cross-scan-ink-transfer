#!/usr/bin/env bash
# C. Against the human ink labels (inklabels 20260918, dense patch = control line 2).
#    Published 2.399 sv AUC 0.90; March scan, mesh as published AUC 0.76; flattened AUC 0.73-0.75.
source runs/env.sh; mkdir -p out; M=$XSCAN_DATA/mesh_20260319133554
python src/fetch_sv.py $SV24 out/l2_pubsv 14400 15600 22080 25440
python src/infer_crop.py out/l2_pubsv --offsets 0
python src/render_crop.py $M $VOL_MARCH out/l2_march --rows 724 784 --cols 1109 1277 --layers 161 --level 1 --oversample 2 --smooth-normals 1.0
python src/refine_surface.py out/l2_march --out-layers 121
python src/infer_crop.py out/l2_march --offsets 0
python src/infer_crop.py out/l2_march_flat --offsets -24 0 24
python src/label_eval.py out/l2_pubsv_ink_o+0.npy 1800 2760
python src/label_eval.py out/l2_march_ink_o+0.npy 1800 2760 48
for o in -24 +0 +24; do python src/label_eval.py out/l2_march_flat_ink_o$o.npy 1800 2760 48; done
