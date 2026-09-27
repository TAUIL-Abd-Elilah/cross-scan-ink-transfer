#!/usr/bin/env bash
# C2. Human-label benchmark of the March-2026 2.4 um scan (PHerc0139 20260319133554) on all 8 PHerc0139 segments
#     that have human ink labels. Per segment: densest labelled 2.9 x 8.2 mm window; reference = published 2.399 um
#     surface volume; March scan = mesh as published, then flattened + 5 depth windows (-96..+96 um); AUC vs labels.
#     ~1.2 GB of CT per segment. Results: results/C2_march_scan_8_segments.json.
source runs/env.sh; export MARCH_BENCH_DATA=out/march_bench; cd src
SEGS="20260112000000-w043_2026011217 20250108000005-w030_2025010818 20250831000000-w040_2025083102 20260108000000-w041_2026010816 20260115000000-w044_2026011522 20260126000000-w045_2026012619 20260302000000-w039_2026030210 20260317000000-w035_2026031718"
python march_bench.py $SEGS
python march_bench_labelfree.py $SEGS
python march_bench_local.py $SEGS
