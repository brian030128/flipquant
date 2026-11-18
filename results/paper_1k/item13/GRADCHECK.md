# Item 13: Appendix F.1 (the tile-gradient check) at the 1k calibration (Mistral-7B, 16x64)

`calibration/gradcheck.py`, a re-implementation of F.1 from the paper's text (the original scripts were never pushed), validated by its 256 run against the published results (step 0). KL changes x 1e-3 nats, paired over the sequences, ± 2 SE. Run from flipquant alone (`python -m experiments.paper_1k.run item13`).

## The 256 run (step 0)

6,815,744 tiles; the map flips 75,906. In-sample: the 256 fit windows; held-out: `512:64:128` (256 windows, disjoint True). Baseline KL against the trainer's epoch 0: relative difference 0.0.

| score | held-out | in-sample | published held-out | published in-sample |
|---|---|---|---|---|
| g_u | +0.73 [+0.41, +0.90] | +0.91 [+0.75, +0.97] | +0.71 [+0.37, +0.92] | +0.90 [+0.73, +0.98] |
| act | -0.45 [-0.81, -0.02] | -0.31 [-0.69, +0.14] | -0.18 [-0.59, +0.30] | -0.41 [-0.76, +0.04] |
| weight_error | +0.08 [-0.43, +0.52] | +0.17 [-0.33, +0.57] | +0.05 [-0.41, +0.48] | -0.17 [-0.63, +0.27] |

Groups beyond 2 SE: held-out 6 (6 with g_u's sign), in-sample 10 (10). Middle bins pooled: insample -0.15 ± 0.44; heldout -0.11 ± 0.43.

| k | predicted | held-out | ratio | in-sample | ratio |
|---:|---:|---:|---:|---:|---:|
| 10 | -1.349 | -1.289 ± 0.593 | +0.96 | -1.165 ± 0.581 | +0.86 |
| 30 | -2.647 | -1.633 ± 0.753 | +0.62 | -2.704 ± 0.684 | +1.02 |
| 100 | -4.931 | -4.179 ± 0.809 | +0.85 | -4.549 ± 0.869 | +0.92 |
| 300 | -8.062 | -5.505 ± 0.847 | +0.68 | -6.128 ± 0.992 | +0.76 |
| 1,000 | -13.566 | -7.426 ± 1.219 | +0.55 | -8.432 ± 1.253 | +0.62 |
| 3,000 | -21.037 | -7.744 ± 1.470 | +0.37 | -9.562 ± 1.517 | +0.45 |
| 10,000 | -33.188 | -5.945 ± 1.745 | +0.18 | -9.008 ± 1.736 | +0.27 |
| 30,000 | -50.807 | -1.230 ± 1.845 | +0.02 | -5.021 ± 1.878 | +0.10 |
| 75,906 | -75.342 | +7.197 ± 2.067 | -0.10 | +2.480 ± 2.011 | -0.03 |
| 100,000 | -85.589 | +9.812 ± 2.110 | -0.11 | +5.493 ± 2.072 | -0.06 |
| 300,000 | -149.904 | +28.376 ± 2.581 | -0.19 | +26.201 ± 2.755 | -0.17 |

The map: predicted -41.13; measured held-out -17.51 ± 1.68 (ratio 0.43), in-sample -26.06 ± 1.95 (ratio 0.63). Its tiles under g_u: 19,129 in the top 75,906, median rank 3.2 %, 130 with g_u > 0.

## The 1k run

6,815,744 tiles; the map flips 359,840. In-sample: the 1024 fit windows; held-out: `2048:448:128` (256 windows, disjoint True). Baseline KL against the trainer's epoch 0: relative difference 0.0.

| score | held-out | in-sample |
|---|---|---|
| g_u | +0.72 [+0.40, +0.90] | +0.88 [+0.69, +0.97] |
| act | -0.43 [-0.76, -0.01] | -0.46 [-0.78, -0.04] |
| weight_error | -0.27 [-0.69, +0.18] | -0.19 [-0.62, +0.27] |

Groups beyond 2 SE: held-out 10 (10 with g_u's sign), in-sample 11 (11). Middle bins pooled: insample -0.13 ± 0.23; heldout -0.31 ± 0.40.

| k | predicted | held-out | ratio | in-sample | ratio |
|---:|---:|---:|---:|---:|---:|
| 10 | -1.368 | -1.079 ± 0.496 | +0.79 | -0.996 ± 0.335 | +0.73 |
| 30 | -2.759 | -2.330 ± 0.723 | +0.84 | -2.484 ± 0.348 | +0.90 |
| 100 | -5.207 | -3.863 ± 0.859 | +0.74 | -4.052 ± 0.440 | +0.78 |
| 300 | -8.467 | -5.167 ± 1.052 | +0.61 | -5.744 ± 0.514 | +0.68 |
| 1,000 | -14.008 | -6.570 ± 1.160 | +0.47 | -7.078 ± 0.574 | +0.51 |
| 3,000 | -21.149 | -7.035 ± 1.504 | +0.33 | -7.744 ± 0.722 | +0.37 |
| 10,000 | -32.260 | -4.427 ± 1.710 | +0.14 | -5.743 ± 0.828 | +0.18 |
| 30,000 | -46.942 | +2.928 ± 1.895 | -0.06 | +1.536 ± 0.937 | -0.03 |
| 100,000 | -72.253 | +24.493 ± 2.529 | -0.34 | +21.468 ± 1.147 | -0.30 |
| 300,000 | -113.915 | +60.748 ± 3.927 | -0.53 | +56.347 ± 1.632 | -0.49 |
| 359,840 | -123.708 | +69.419 ± 4.272 | -0.56 | +64.662 ± 1.744 | -0.52 |

The map: predicted -53.65; measured held-out -22.46 ± 1.66 (ratio 0.42), in-sample -33.23 ± 1.09 (ratio 0.62). Its tiles under g_u: 91,729 in the top 359,840, median rank 13.5 %, 25,979 with g_u > 0.

## Step 0

The 256 run against the published results: own checks True; baseline KL True; the map within the published 2 SE {'heldout': True, 'insample': True}; g_u's Spearman inside the published intervals {'heldout': True, 'insample': True} — **pass**.

