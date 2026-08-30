# MATLAB package

Ports of the analysis in `analysis/` so the heavy Monte Carlo can run on a real
machine. Same conventions and same fairness rules as `analysis/THEORY.md`.

| file | what it does | runtime |
|---|---|---|
| `nf_lib.m` | shared model, analytic formulas, estimator | library |
| `run_theory_checks.m` | verifies propositions 1-4 | seconds |
| `run_threshold_mc.m` | outlier threshold and RMSE-vs-CRB curves | tens of minutes at `NR=150` |

## What to run first

`run_theory_checks.m`. It should reproduce: zooming-map error under ~1.1%,
`centre/eta` and `width/eta` constant across subcarriers, measured/predicted
width ratio in 0.7-1.15, and pilot saving 1.9x at `dec=8` / 4x at `dec=4` for
`theta=0.6`.

## What is actually worth your compute

`run_threshold_mc.m`, and specifically the first scheme. The Python sweep never
bracketed the full-codebook threshold - it bottomed out at 19.4 dB with zero
outliers, so the constant that anchors the threshold predictor is an upper
bound. `A0list` here starts lower to pin it. Please report where it breaks.

Raise `NR` to 1000 if you can leave it running; 150 is enough to see the
threshold but the RMSE curves are still visibly noisy near it.

## Reference numbers to compare against

At `theta=0.6`, `r=12` m, 10%-outlier threshold in peak SNR:

| scheme | pilots | E_eff | threshold |
|---|---|---|---|
| narrowband, full codebook | 256 | 4.59 | <= 19.4 dB (not bracketed) |
| narrowband, every 4th beam | 64 | 1.38 | 23.4 dB |
| wideband, every 4th beam | 64 | 8.85 | 14.0 dB |

The 9.4 dB gap between the two 64-pilot rows is the headline result.

## One caveat on the CRB used here

`crb_range` in `run_threshold_mc.m` is the high-SNR Gaussian approximation to
the Rician CRB, not the exact Rician bound that `analysis/` computes by
numerical integration. The two agree above threshold, which is the only place
the CRB describes anything; below threshold neither is meaningful.
