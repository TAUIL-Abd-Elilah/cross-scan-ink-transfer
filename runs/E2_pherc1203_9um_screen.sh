# 9.36 um screen of the four PHerc1203 auto_grown segments on the ELIGIBLE 9.362 um volume:
# render 28 layers (like the published native renders), then ink_9um s42-75k vs ft_c 6k.
source runs/env.sh
P=python
V=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr
D=out/1203  # seg_<id> meshes from E_pherc1203_screen.sh
O=out/1203/screen9; mkdir -p $O
for s in 20251005230830031 20251005231446965 20251005221856743 20250925223153537; do
  read R C < <($P -c "import tifffile;x=tifffile.imread('$D/seg_$s/x.tif');print(x.shape[0]-1,x.shape[1]-1)")
  for try in 1 2 3; do
    $P src/render_crop.py $D/seg_$s $V $O/$s --rows 0 $R --cols 0 $C --layers 28 --smooth-normals 0.5 --cache-chunks 2500 2>&1 | tail -n 1
    [ -f $O/$s.npy ] && break; sleep 60
  done
  $P -c "
import numpy as np, zarr
a=np.load('$O/$s.npy'); z=zarr.open_array('$O/$s.zarr', mode='w', shape=a.shape, chunks=(a.shape[0],128,128), dtype='u1', zarr_format=2, compressor=None, fill_value=0); z[:]=a; print('zarr', a.shape)"
  rm -f $O/$s.npy
  for ck in "ink9um_s42_075k=$INK9UM_CKPTS/hybrid_3d2d-seed42/step-075000.pth" "ft_b_6k=models/ink9um_native0139_ft6k.pth"; do
    n=${ck%%=*}; c=${ck#*=}
    PYTHONPATH="$INK9UM_SRC" $P -m vesuvius.ink_detection.inference.infer $O/$s.zarr "$c" $O/${s}_$n.tif --overlap 0.5 --blend-mode hann --batch-size 8 --direction both --no-compile > $O/${s}_$n.log 2>&1
    ls $O/${s}_$n*.tif
  done
done
echo SCREEN9_DONE
