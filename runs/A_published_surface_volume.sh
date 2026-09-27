#!/usr/bin/env bash
# A. Our inference wrapper on the PUBLISHED 2.399 um surface volume reproduces the published
#    canonical prediction (control line 1: px rows 14640:16080, cols 11320:17000). Result: r 0.93 at o0.
source runs/env.sh; mkdir -p out
python src/fetch_sv.py $SV24 out/l1_pubsv 14640 16080 11320 17000
echo '{"rows":[732,804],"cols":[566,850],"upsample":20,"grid":"published"}' > out/l1_pubsv.json
python src/infer_crop.py out/l1_pubsv --offsets -20 -10 0 10 20
for o in -20 -10 +0 +10 +20; do python src/compare_ctrl.py out/l1_pubsv out/l1_pubsv_ink_o$o.npy; done
