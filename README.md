# sar-attention-seg

Attention-gated encoder–decoder segmentation of dual-polarisation C-band SAR stacks with change-detection weak supervision.

## Pipeline

| Stage | Module | Operations |
|---|---|---|
| Ingest | `src/gee_ingest.js` | GRD σ⁰ (VV/VH) → Refined Lee despeckling → volume-model radiometric terrain flattening (γ⁰_f) → per-orbit temporal median reference → log-ratio change detection with θ_loc-weighted polarimetric fusion → co-registered 20/100/500 m tensor export |
| Learn | `src/seg_pipeline.py` | weak labels via threshold + auxiliary-covariate consistency masking → spatially blocked train/val/test partition → Attention U-Net (additive attention gates on 4 skip connections, BN-ReLU, 16→256 filters) → stochastic 256² crops, flip augmentation, AdamW, BCE + soft-Dice composite loss, early stopping → full-scene inference with 16-px-aligned padding → OA, F1, IoU, κ with non-parametric bootstrap CIs, 8-neighbourhood isolation metric, cross-sensor inclusion rate |
| Series | `src/aux_series.py` | per-pixel binary time series → first-passage transition dates → stratified Theil–Sen / Mann–Kendall trend estimation, robust elevation-gradient regression |

## Run

```bash
python3 src/seg_pipeline.py prep  --chips CHIPS --modis REF.tif --out results
python3 src/seg_pipeline.py train --chips CHIPS --modis REF.tif --out results
python3 src/aux_series.py IN_DIR OUT_DIR
```

`gee_ingest.js` runs in the Earth Engine Code Editor; grid, extent and asset constants are left blank.

## Dependencies

Earth Engine, `gee_s1_ard` (Mullissa et al., 2021), Python ≥ 3.11, PyTorch, NumPy, SciPy, pandas, rasterio.

## Status

Accompanies a manuscript under review. Full documentation follows publication.

## Licence

MIT © 2026 Polina Lemenkova
