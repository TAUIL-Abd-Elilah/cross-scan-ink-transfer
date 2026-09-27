#!/usr/bin/env bash
# F. Fine-tune ink_9um on native 9.362 um PHerc0139 renders with canonical 2.4 um labels; test on
#    held-out PHerc0139 (human labels) and on unseen scrolls PHerc0841 (human) / PHerc0343P.
# Needs: INK9UM_SRC (villa checkout: vesuvius/src with vesuvius.ink_detection), INK9UM_CKPTS
#        (huggingface.co/scrollprize/ink_9um), DIST9_DATA (~30 GB of native renders + labels).
source runs/env.sh; cd src/dist9
TEST="20250108000005-w030_2025010818 20260126000000-w045_2026012619"
TRAIN="20250108000000-w025_2025010863 20250108000001-w026_2025010854 20250108000002-w027_2025010845 20250223000000-w059_2025022312 20251226000000-w055_2025122611 20260115000001-w056_2026011514 20260127000000-w057_2026012713 20260130000000-w031_2026013019 20260203000000-w032_2026020303 20260206000000-w042_2026020613 20260206000001-w047_2026020613 20260206000002-w054_2026020617 20260112000000-w043_2026011217"
XS="PHerc0841/20260220213127-w00 PHerc0841/20260220214732-auto_grown_20260220144552896 PHerc0841/20260221022814-auto_grown_20260220174252405 PHerc0343P/20250511003658-tifxyz PHerc0343P/20250902170435--5_b2 PHerc0343P/20250902170441--4_b2 PHerc0343P/20250902170447--3_b2 PHerc0343P/20250902171202--2_b2 PHerc0343P/20250902171204--1_b2 PHerc0343P/20250904233748-0_b2 PHerc0343P/20250905172054-1_b2"
XSD=$(for x in $XS; do echo -n "${x%%/*}__${x#*/} "; done)
python prep_pair.py $TEST $TRAIN $XS                 # native renders -> local zarr, 2.4 um labels -> native grid
python align_pair.py $TEST $TRAIN $XSD               # stretch to the native canvas + high-pass NCC offset
python fix_offset.py $TRAIN                          # run b: the systematic offset (-8, -7) for all 0139 labels
python ft9.py --train $TRAIN --label canon_fx.npy --out runs/ft_b --steps 6000 --batch 24 --lr 1e-3 --save-every 2000
C=$INK9UM_CKPTS
python eval9.py --segments $TEST --out results_eval.json --ckpt ink9um_s42_075k=$C/hybrid_3d2d-seed42/step-075000.pth ft_b_6k=runs/ft_b/ft-006000.pth
python eval9.py --segments $XSD --out results_xscroll.json --ckpt ink9um_s42_075k=$C/hybrid_3d2d-seed42/step-075000.pth ft_b_6k=runs/ft_b/ft-006000.pth
