*Experiments completed: 2026-04-07 to 2026-04-11*

**Note:** Random split results differ slightly from the published paper ([arXiv:2506.20132](https://arxiv.org/abs/2506.20132)). In particular, the random initialized baseline improved (RMSE 23.61 to 21.88), narrowing the gap with the pretrained model from ~20% to ~12%. The exact cause is unknown but may include differences in random seeds, library versions, or a fix to checkpoint resume logic.

# Results (random vs spatial)

## Table 1: Overall Results

| Category | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Pretrained | 19.14 | 12.80 | 0.71 | 24.90 | 17.68 | 0.48 |
| Random init | 21.88 | 15.02 | 0.63 | 26.34 | 18.81 | 0.41 |
| Monthly pred | 33.66 | 25.38 | 0.12 | 32.33 | 25.06 | 0.12 |

## Table 2: Season Breakdown

| Season | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 19.14 | 12.80 | 0.71 | 24.90 | 17.68 | 0.48 |
| Winter | 14.72 | 10.06 | 0.79 | 18.77 | 12.90 | 0.41 |
| Spring | 23.34 | 15.71 | 0.68 | 31.88 | 22.83 | 0.34 |
| Summer | 19.87 | 13.50 | 0.66 | 25.76 | 18.79 | 0.42 |
| Autumn | 13.06 | 9.18 | 0.73 | 15.60 | 12.07 | 0.57 |

## Table 3: Land Cover Breakdown

| Land Cover | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 19.14 | 12.80 | 0.71 | 24.90 | 17.68 | 0.48 |
| Trees | 18.07 | 12.24 | 0.68 | 25.49 | 18.13 | 0.39 |
| Grass | 20.93 | 14.10 | 0.71 | 24.39 | 17.30 | 0.55 |
| Shrub | 19.10 | 12.13 | 0.75 | 24.76 | 17.24 | 0.49 |
| Built-up | 17.22 | 11.61 | 0.76 | 20.75 | 15.99 | -0.37 |
| Bare / Sparse | 19.14 | 13.48 | 0.82 | 25.33 | 19.09 | 0.68 |

Built-up and Bare/Sparse have limited site diversity (21 and 15 sites respectively), which may reduce the reliability of spatial split metrics for these classes.

## Table 4: Elevation Breakdown

| Elevation | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 19.14 | 12.80 | 0.71 | 24.90 | 17.68 | 0.48 |
| 0-500m | 18.51 | 11.68 | 0.73 | 19.37 | 14.05 | 0.53 |
| 500-1000m | 18.22 | 12.30 | 0.76 | 23.95 | 16.70 | 0.56 |
| 1000-1500m | 21.60 | 14.50 | 0.73 | 30.10 | 20.45 | 0.44 |
| 1500-2000m | 19.09 | 13.69 | 0.74 | 25.91 | 19.34 | 0.49 |
| 2000-2500m | 19.57 | 12.82 | 0.60 | 24.67 | 18.40 | 0.36 |
| 2500-3000m | 16.41 | 10.73 | 0.46 | 21.87 | 16.17 | 0.03 |
| 3000-3500m | 15.70 | 11.50 | 0.20 | 23.85 | 19.38 | -0.10 |

## Table 5: Shape Ablations

| H, W | T | P | Random RMSE | Random MAE | Random R2 | Spatial RMSE | Spatial MAE | Spatial R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 12 | 16 | 19.14 | 12.80 | 0.71 | 24.90 | 17.68 | 0.48 |
| 32 | 6 | 16 | 19.13 | 12.75 | 0.72 | 25.58 | 17.66 | 0.45 |
| 32 | 3 | 16 | 20.10 | 13.79 | 0.69 | 26.27 | 17.82 | 0.42 |
| 16 | 12 | 16 | 19.99 | 13.75 | 0.69 | 25.72 | 17.72 | 0.44 |
| 8 | 12 | 8 | 19.76 | 13.42 | 0.70 | 25.27 | 17.85 | 0.46 |
| 1 | 12 | 1 | 20.72 | 13.95 | 0.67 | 24.66 | 17.04 | 0.49 |

## Table 6: Data Ablations

| Excluded | Rnd PT RMSE | Rnd PT MAE | Rnd PT R2 | Rnd RI RMSE | Rnd RI MAE | Rnd RI R2 | Spt PT RMSE | Spt PT MAE | Spt PT R2 | Spt RI RMSE | Spt RI MAE | Spt RI R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | 19.14 | 12.80 | 0.71 | 21.88 | 15.02 | 0.63 | 24.90 | 17.68 | 0.48 | 26.34 | 18.81 | 0.41 |
| ERA5 | 18.98 | 12.64 | 0.72 | 26.43 | 18.77 | 0.46 | 25.53 | 17.79 | 0.45 | 27.13 | 20.25 | 0.38 |
| loc. | 19.27 | 13.37 | 0.71 | 24.23 | 16.96 | 0.54 | 25.96 | 17.97 | 0.43 | 27.86 | 19.58 | 0.35 |
| S1 | 19.39 | 13.23 | 0.71 | 24.66 | 17.14 | 0.53 | 25.43 | 18.52 | 0.45 | 27.29 | 20.22 | 0.37 |
| S2 | 19.33 | 12.78 | 0.71 | 25.17 | 18.31 | 0.51 | 25.87 | 17.73 | 0.44 | 27.49 | 19.35 | 0.36 |
| TC | 18.90 | 12.33 | 0.72 | 22.32 | 15.34 | 0.61 | 26.91 | 18.74 | 0.39 | 26.68 | 18.81 | 0.40 |
| SRTM | 19.40 | 13.10 | 0.71 | 25.29 | 17.74 | 0.50 | 25.58 | 17.79 | 0.45 | 26.17 | 18.67 | 0.42 |

Rnd = Random split, Spt = Spatial split, PT = Pretrained, RI = Random Initialized

