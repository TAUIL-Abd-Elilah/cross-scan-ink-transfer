# Edit these, then: source runs/env.sh
export XSCAN_DATA=${XSCAN_DATA:-data}          # runs/fetch_refs.sh output
export XSCAN_CACHE=${XSCAN_CACHE:-cache}        # CT chunk cache (a few GB)
export VILLA_INFERENCE_DIR=${VILLA_INFERENCE_DIR:-villa/ink-detection/optimized_inference}   # ScrollPrize/villa checkout
export CANON_CKPT=${CANON_CKPT:-models/r152_3ddec_v2_l5_epoch13.ckpt}   # huggingface.co/scrollprize/ink_canonical_2um
export BUCKET=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
export SV24=$BUCKET/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr
export SV93=$BUCKET/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr
export VOL_MARCH=$BUCKET/PHerc0139/volumes/20260319133554-2.403um-0.2m-77keV-masked.zarr
