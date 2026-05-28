# Data workspace

This directory is reserved for local training and evaluation data.

Raw planetary products can be large, so `data/raw/` and `data/processed/`
are intentionally ignored by git. Keep durable provenance in
`data/manifests/` instead.

Current quasi-real intake:

- `data/raw/quasi_real/lunar_south_pole/lro_lola_gdr_875s_20m/`
  - LRO LOLA south-pole gridded data products.
  - Polar stereographic projection centered on the lunar south pole.
  - Latitude range: -90 to -87.5 degrees.
  - Resolution: 20 meters per pixel.
  - Purpose: terrain/elevation and observation-density inputs for candidate
    list policy training fixtures and evaluation scenarios.

Generated smoke outputs belong under `data/processed/` and should remain local.
The current pipeline records provenance in rollout transition metadata and in
experiment dataset summaries so quasi-real samples are not confused with real
mission labels.
