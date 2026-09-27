# cross-scan-ink-transfer

Several scrolls, including the prize-eligible PHerc1203 and PHerc0846A, now have **March-2026 2.4 µm scans**.
Can the canonical 2 µm ink model (`scrollprize/ink_canonical_2um`) read them, and what does it take?

**Headline (section C2):** on all 8 PHerc0139 segments that have human ink labels, the canonical model reads the
March-2026 scan at mean AUC **0.756** with the published meshes, against 0.872 on the scan the labels were drawn
on. The ink sits up to 100 µm off those meshes. A **label-free depth pick** (deep render, flatten, keep the most
confident of 5 depth windows) raises it to **0.803**, within 0.007 of choosing the depth with the labels. On
w044 it goes from 0.70 to 0.90.

Also: 4.8 µm reads cost a fifth of the download for most of the signal (B); the canonical model does not read the
real 9.362 µm scan (D); a 5.8 cm² screen of PHerc1203 at 2.4 µm and a ~30 cm² screen at 9.36 µm are negative (E);
and `ink_9um` fine-tuned on native 9.362 µm data reads an unseen scroll better, on AUC and on Scheirer's
letter-scale score, though not yet legibly (F).

## Results

**Line 1** (published-grid rows 732-804, cols 566-850), scored against the published canonical
prediction (model vs model), r at the best alignment within +-24 ds8 px:

| input | r | null (other text, max) |
|---|---|---|
| A. published 2.399 µm surface volume (scan 20260102150214), our wrapper | **0.93** | 0.43 |
| A. same, window moved +-10 / +-20 layers | 0.85-0.88 / 0.52-0.81 | |
| B1. March scan 20260319133554, published on-scan mesh as is | **0.11** | 0.09 |
| B2. March scan, deep render + flattening, window 46 layers (110 µm) off the mesh | **0.76** | 0.41 |
| B3. March scan, same from **4.8 µm** data (level 1, 2x oversampled), 278 vs ~1,400 chunks | **0.73** | |
| D. line-1 2.399 µm data, 2x2x2 mean-pooled then back to the 2.4 grid | 0.885 | |
| D. same, 4x4x4 (synthetic 9.6 µm) | 0.66 | |
| D. **real 9.362 µm** surface volume (scan 20250728140407), x3.9 to the 2.4 grid | **0.32** (p99.9 of all positions 0.27) | |

**Line 2** (published-grid rows 720-780, cols 1104-1272), scored against the **human ink labels**
(`ink-labels/.../20260918`), AUC of ink probability at labelled ink vs labelled background:

| input | AUC | r | null AUC (mean / max) |
|---|---|---|---|
| published 2.399 µm surface volume | **0.90** | 0.68 | 0.45 / 0.69 |
| March scan, published on-scan mesh as is (4.8 µm read) | **0.76** | 0.49 | 0.45 / 0.56 |
| March scan, flattened, window at 0 / -24 layers | 0.73 / 0.75 | 0.41 / 0.43 | 0.43-0.46 / <=0.59 |

Figures (top = this pipeline, bottom = published prediction): `figures/`.

**C2. The same test on all 8 human-labelled PHerc0139 segments.** Every PHerc0139 segment with published human
ink labels (w030, w035, w039, w040, w041, w043, w044, w045) also has a mesh on the March-2026 scan. For each, we
took the densest labelled 2.9 x 8.2 mm window (about 1.7 cm² of labelled text in total) and scored the canonical
model against the human labels (AUC, labelled ink vs unlabelled papyrus within 2 mm of it):

| segment | 2.399 µm scan (labels drawn here) | March scan, mesh as published | March scan, flattened + label-free depth pick | (depth) | oracle best of 5 depths |
|---|---|---|---|---|---|
| w043 | 0.920 | 0.768 | 0.772 | -48 µm | 0.772 |
| w030 | 0.936 | 0.923 | 0.907 | -48 µm | 0.907 |
| w040 | 0.872 | 0.812 | 0.804 | 0 | 0.804 |
| w041 | 0.888 | 0.838 | 0.802 | +48 µm | 0.824 |
| **w044** | 0.966 | **0.696** | **0.903** | -96 µm | 0.903 |
| w045 | 0.840 | 0.613 | 0.700 | -96 µm | 0.701 |
| **w039** | 0.776 | **0.643** | **0.782** | -96 µm | 0.782 |
| w035 | 0.782 | 0.752 | 0.753 | +48 µm | 0.786 |
| **mean** | **0.872** | **0.756** | **0.803** | | 0.810 |

- The **label-free depth pick** uses no labels. It renders 161 layers (about ±190 µm) along the published mesh's
  smoothed normals, flattens the render onto the local papyrus layering (`src/refine_surface.py`), runs the
  model at 5 depth windows (-96 to +96 µm), and keeps the window whose ink map has the most pixels above 0.5.
- It recovers **40 % of what the published meshes lose** (0.756 -> 0.803 of 0.872) and lands within 0.007 of
  choosing the depth with the labels. On 3 of 8 segments the ink sits 50-100 µm off the mesh and the gain is
  large: w044 0.70 -> 0.90 (`figures/C2_march_scan_w044.png`), w039 0.64 -> 0.78, w045 0.61 -> 0.70. Where
  the mesh was already right it costs at most 0.036 (w041).
- A pixel-wise maximum over the depth windows does not help (mean 0.753): taking one depth is better than
  merging several.
- Reading the March scan with 4.8 µm data on the 2.4 µm grid costs about 1.2 GB of CT per cm².
- Code: `src/march_bench.py`, `src/march_bench_labelfree.py`, `runs/C2_march_scan_benchmark.sh`. All numbers:
  `results/C2_march_scan_8_segments.json`.
- `w043` may be in the canonical model's training data and the labels were drawn on the 2.399 µm scan, so the
  reference column is an upper bound. The comparison between the three March columns is the point.

**E. A negative on an eligible scroll: PHerc1203, March-2026 2.4 µm scan.** The same pipeline,
screening about **5.8 cm²** of the cleanest sheet surface we could find, found **no text**:

- The 2.4 µm scan was registered to the eligible 9.362 µm scan: outline descriptors, then a similarity
  transform fitted on 38 µm blocks. NCC is 0.916, and fits at two heights agree to 0.7 voxel at 9.36 µm.
- Four published `auto_grown` segments were mapped into the 2.4 µm frame. Each whole segment was
  surveyed at 19 µm to find where the flattened surface stays on one sheet (straight fibre crosshatch)
  rather than folds.
- 14 tiles were screened at 15 depths (-84 to +84 layers) from 4.8 µm data.
- The known-ink control reaches a best 2 mm window of 0.68-0.80 of pixels above 0.5. On PHerc1203, 8 tiles
  stay at or below 0.29. The 6 flagged tiles were checked by eye, and every flag is structure:
  - a bright cracked crust with a bubble cluster, which scored 0.97 when re-rendered at full resolution
    and is the clearest false positive (`figures/E_pherc1203_false_positive_crust_fullres.png`);
  - creases, mesh edges, and one isolated 3.4 mm chain along a vertical fibre strip.

  Per-tile numbers and verdicts are in `results/pherc1203_screen_results.json`.
- **On the eligible 9.362 µm volume itself** the same four segments were rendered whole (about 30 cm², 28
  layers) and read with both the released `ink_9um` and the fine-tuned model from section F, in both
  face directions (`runs/E2_pherc1203_9um_screen.sh`,
  `figures/E_pherc1203_9362um_seg031_released_vs_finetuned.png`). Neither shows text: only speckle,
  strongest over folds.
- What this does and does not say: with this model, these 5.8 cm² show no readable ink. It does not say
  PHerc1203 is blank. High ink scores on this scroll come from crust and folds, so **shape, not score,
  decides**.

**F. Training on real 9 µm data does transfer better to unseen scrolls (partly).** D shows that 2.4 µm
data degraded to 9 µm overstates what a model can read from a real 9 µm scan. The public
`scrollprize/ink_9um` was trained mostly on such degraded data, plus 5 native 9.362 µm segments.
PHerc0139 has about 38 segments with both a native 9.362 µm render and a canonical 2.4 µm prediction.

We fine-tuned `ink_9um` (hybrid_3d2d seed 42, step 75k):
- **Training data:** 12 native segments it never saw (13 in run b), with the canonical 2.4 µm
  prediction as the label.
- **Label transfer:** labels were resized onto the native canvas. The two canvases differ by 0.3-0.4 %
  from a pure 9.362/2.399 scale, so a shift alone leaves a drift across the segment. They were then
  aligned by high-pass cross-correlation; the offset is systematic, about (-8, -7) px, on every held-out
  segment.
- **Recipe and checkpoints:** the released model and normalization are kept unchanged, and checkpoints
  load in the stock CLI.
- **Test data:** never trained on. Two held-out PHerc0139 segments with human labels (w030, w045), and
  two scrolls absent from `ink_9um`'s training: PHerc0841 (native 9.366 µm, human labels) and PHerc0343P
  (native 8.64 µm, canonical prediction only).

AUC of the 9 µm prediction against human labels (canonical-label AUC for PHerc0343P):

| test | released ink_9um (4 checkpoints) | fine-tuned, run a (4k) / run b (6k) | 2.4 µm teacher |
|---|---|---|---|
| PHerc0139 w030 + w045 (human) | 0.856-0.883 | 0.857 / 0.858 | 0.852-0.870 |
| **PHerc0841, unseen scroll (human)** | **0.734-0.745** | **0.803 / 0.804** | 0.947 |
| PHerc0343P, unseen, 8.64 µm (canonical) | 0.557-0.568 | 0.656 / 0.661 | - |

**Letter-scale check.** AUC and correlation can rise just because a model smooths its output. Chris
Scheirer's [9 µm reader benchmark](https://github.com/ShribyrLabs/vesuvius-reports/tree/main/02-9um-ink-reader-benchmark)
defines a letter-scale score for exactly this: Pearson r after a 48 µm high-pass on both read and key (hp r),
with rolled-key nulls. On his scale the released `ink_9um` scores about 0.035, his fine-tune 0.11, a read where
a person made out four letters 0.076, and run-to-run noise is 0.005. We ran his score, unchanged, on our tests
(`src/dist9/hp_eval.py`, key = canonical 2.4 µm prediction on the native grid):

| test | released ink_9um s42 / s43 (75k) | fine-tuned run a / b / c | rolled-key null |
|---|---|---|---|
| PHerc0139 w030 + w045 | 0.044 / 0.048 | 0.061 / **0.070** / 0.060 | <= 0.002 |
| **PHerc0841, unseen scroll** | 0.028 / 0.028 | 0.053 / **0.053** / 0.049 | <= 0.005 |

His caveat is that blurring a read raises hp r, so reads only compare at equal blur. We blurred every model's
output by 0, 2 and 4 px and scored all of them on identical pixels (`src/dist9/hp_blur.py`). On PHerc0841, run b
scores 0.059 / 0.094 / 0.125 and the released model 0.030 / 0.042 / 0.054. The released model blurred by 4 px
still scores below the fine-tune unblurred, so the gain is not smoothing
(`results/F_letter_scale_blur_control.json`). Hp r against the sparse human labels is near zero for every model,
so it does not separate them; the human labels are used for AUC above.

- **Within PHerc0139:** no AUC gain (0.858 vs its starting checkpoint's 0.870, inside the released range),
  but a letter-scale gain (hp r 0.070 vs 0.044). The released model already matches its 2.4 µm teacher on AUC there.
- **On the unseen scroll:** a stable gain across three runs: AUC 0.74 -> 0.80 (about 30 % of the gap to the
  teacher) and letter-scale hp r 0.028 -> 0.053, on a scroll absent from both `ink_9um`'s and our training.
  Scheirer reported a similar-sized gain for his fine-tune on another unseen scroll (PHerc0814, 0.035 -> 0.076);
  this is a second, independent scroll with human labels.
- **A second training scroll does not help yet.** Adding PHerc0343P's 8 segments to training (run c)
  gives 0.810 on PHerc0841 against 0.804. The gain plateaus near 0.80-0.81 whatever the data mix.
- **Depth window:** moving the 17-layer window changes human AUC by only +-0.03
  (`results/F_depth_probe.json`).
- **Limits:**
  - PHerc0841's three segments cover the same patch, so this is one labelled region.
  - **Neither model makes those letters legible at 9 µm** (`figures/F_pherc0841_released_vs_finetuned.png`).
  - This is a measured step, not a reading.
- The fine-tuned checkpoint is attached to the release; the code is in `src/dist9/`.

## What it means in practice

1. **The March-2026 2.4 µm scans are readable by the canonical model**: mean AUC 0.76 against human
   labels over 8 segments with the published meshes, versus 0.87 on the scan the labels were drawn on.
2. **The published on-scan meshes can be off the sheet by 50-110 µm** (3 of 8 labelled segments, and line 1,
   where the mesh as published reads nothing, r 0.11). **Always scan depth before calling a region empty.**
   The label-free depth pick in C2 recovers 40 % of the loss with no labels.
3. **4.8 µm reads keep most of the signal** (0.73 vs 0.76) for about a fifth of the download.
4. **The canonical model does not read the real 9.362 µm scan** (0.32, not legible), although the
   synthetic 4x degradation of 2.4 µm data predicts 0.66 (both plain correlation with the published
   prediction). By that measure synthetic degradation overstates 9 µm transfer about 2x, so the gap is
   scan physics, not only resolution. (Scheirer measured the drop from native 2.4 µm to synthetic 9 µm on his
   letter-scale score, 0.95 -> 0.11; that score is stricter. D adds the canonical model run on the real 9.362 µm scan of the same
   segment.) A 9 µm model should be trained on real, registered 9 µm data rather than on degraded
   2.4 µm data, and F shows that this helps on an unseen scroll.

## Caveats

- Sections A-D use one segment and two text lines; C2 extends the March-scan result to all 8 labelled segments.
  `w043` (and possibly other PHerc0139 segments) may be in the canonical model's training data, so the absolute
  numbers are optimistic; the comparisons between scans and settings are the point.
- Line-1 numbers are agreement with another model's prediction, not with ground truth. Human labels
  exist only on line 2.
- The two grids differ slightly (the March mesh is a transformed copy); alignment is searched, and
  the null columns show what the search finds against the wrong text.

## Reproduce

Needs a CUDA GPU (~6 GB), a ScrollPrize/villa checkout (for `ink-detection/optimized_inference`),
the checkpoint from `huggingface.co/scrollprize/ink_canonical_2um`, and a few GB of CT downloads.

```bash
source runs/env.sh           # set VILLA_INFERENCE_DIR, CANON_CKPT, cache/data dirs
bash runs/fetch_refs.sh      # meshes, published ds8 prediction, labels (~25 MB)
bash runs/A_published_surface_volume.sh
bash runs/B_march_scan_line1.sh
bash runs/C_human_labels.sh
bash runs/C2_march_scan_benchmark.sh   # all 8 labelled segments (~10 GB of CT)
bash runs/D_resolution.sh    # uses out/l1_pubsv.npy from A
bash runs/E_pherc1203_screen.sh   # registration, mesh mapping, one screening tile
bash runs/F_native9um_finetune.sh # native 9 um fine-tune + held-out and cross-scroll tests (~30 GB)
```

`results/` holds the numbers above as produced; `src/` is the code that produced them:
`render_crop.py` (sub-rectangle renderer with normal smoothing, pyramid level and oversampling,
built on the validated `render_tifxyz.py`), `refine_surface.py` (cross-correlation dip flattening),
`infer_crop.py` (the villa `resnet3d-152-3d-decoder` recipe: 62 layers, clip 200, 256 tiles, Hann
blend), `compare_ctrl.py`, `compare_wide.py`, `label_eval.py`, `ink_stats.py`, `fetch_sv.py`;
for section E, `zfetch.py`, `register_outline.py`, `register_fine.py`, `check_consistency.py`,
`map_mesh.py`, `sheet_depths.py`, `flag_view.py`.

MIT licence. Data: Vesuvius Challenge, CC BY-NC 4.0.
Developed with AI assistance under my direction.
