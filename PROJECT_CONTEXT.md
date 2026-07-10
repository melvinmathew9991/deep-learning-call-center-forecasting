# PROJECT_CONTEXT.md

This file is a standalone project-context summary — paste it (or point to it) at the start of a session so Claude can pick up this project without re-analyzing it from scratch. It is a plain file, not auto-loaded by Claude Code (that's what `CLAUDE.md` would be, but this was renamed away from that at the user's request).

## Project overview

Forecasts monthly call center volume (Healthcare sector) using four deep learning architectures — MLP, CNN, LSTM, CNN-LSTM — on a univariate time series. The dataset (`data/CallCenterData.xlsx`) has 132 monthly points (Jan 2010–Dec 2020) across 5 sector columns, but only the `Healthcare` column is ever modeled; the other four (Telecom/Banking/Technology/Insurance) are plotted for visual comparison only, never trained on.

## Commands

Setup:
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; .venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
```

Run the full pipeline (must be run with `src/` as the working directory — `ml_pipeline` is imported as a plain top-level package, not installed):
```bash
cd src
python engine.py
```
This loads the data, plots each sector to `output/plots/`, builds lagged training windows, trains all 4 models, saves them in native Keras format (`output/models/*.keras`), saves prediction plots (`output/plots/*.png`), and writes a MAE/RMSE/MAPE comparison to `output/model_comparison.csv`.

`notebooks/DeepLearning.ipynb` mirrors `src/` exactly (same splits, architectures, seeding, metrics) as a self-contained notebook for Jupyter/Colab — if you change the modeling logic in `src/`, make the equivalent change there too, or they will drift out of sync again (see history below).

There is no test suite, linter, or CI configured in this project.

## Architecture

- **`src/engine.py`** — orchestrates the whole pipeline: load xlsx → build lag windows (5 lags `t-4..t` → 1-step-ahead target `Y`, via a manual loop, not pandas `.shift()`) → chronological train/valid/test split → fit `MinMaxScaler` on train only → train all 4 models sequentially → print/save the metrics comparison.
- **`src/ml_pipeline/utils.py`** — shared across all model files: path constants (`DATA_DIR`, `OUTPUT_DIR`, `PLOTS_DIR`, `MODELS_DIR`, derived from `PROJECT_ROOT` so nothing depends on string-literal relative paths), `to_series()` (reshape to `(samples, timesteps, 1)` for Conv1D/LSTM input), `evaluate()` (MAE/RMSE/MAPE), `plot_predictions()` (actual vs. predicted with legend), `seed_everything()` (fixes Python/NumPy/TF RNGs — called once at the top of `engine.py`, before any model work).
- **`src/ml_pipeline/{mlp,cnn_model,lstm_model,cnn_lstm_model}.py`** — one class per architecture. Each does build → train → predict → plot → compute metrics inside `__init__` (a side-effecting constructor; this is the established pattern here — keep it consistent rather than refactoring to a separate build/fit/predict API).
- **Each model class creates its own `Adam` optimizer instance.** Never share one optimizer across models/compiles — Keras 3 raises `ValueError: Unknown variable` if an optimizer already bound to one model's variables is reused for another's. This is a real bug that was hit and fixed in the notebook (the original tutorial code shared a single `adam` across all 4 models).
- Splits are chronological throughout: last 20 rows = test, most recent ~10% of the remainder = validation. Don't reintroduce `sklearn.train_test_split`-style shuffling — adjacent lag-windows overlap on 4 of 5 features, so a shuffled split leaks near-duplicate rows across train/valid.
- Scaler is fit on train only, then applied to train/valid/test — never re-fit on test.

## Project history / current state

This started as an unmodified tutorial-style repo with several real bugs, since fixed:
- Validation split was randomly shuffled (leakage across overlapping windows) → now chronological.
- `batch_size=256` was defined but never passed to `.fit()` (silently trained at Keras's default 32) → now wired up in all 4 model files.
- No quantitative metrics, only eyeballed plots → added MAE/RMSE/MAPE (`output/model_comparison.csv`).
- Models were pickled directly (fragile across Keras versions) → now saved via native `model.save(...)` `.keras` format.
- No random seed anywhere → results were not reproducible run-to-run (confirmed LSTM RMSE swinging between 0.39 and 0.64 across identical runs) → fixed via `seed_everything()`.
- The notebook had drifted out of sync with `src/`: worse leakage (scaler fit on the full pre-split data), an `LSTM(90)` vs. `LSTM(50)` architecture mismatch, a wrong data path, and the shared-`Adam`-optimizer bug above (would have crashed on the first non-MLP `.fit()` call under Keras 3). Rewritten to mirror `src/` exactly and validated by actually executing it end-to-end (not just read).
- `output/models/` and `output/plots/` used to rely on committed `.gitkeep` placeholders to exist after a fresh clone (since git doesn't track empty directories, and neither `plt.savefig()` nor `model.save()` create missing parent directories). Replaced with `PLOTS_DIR.mkdir(parents=True, exist_ok=True)` / `MODELS_DIR.mkdir(...)` in `utils.py` — more robust, and survives a local `rm -rf output/` too. `.gitkeep` files removed; `.gitignore` now just excludes `output/` wholesale.

**Open / not yet decided**: `data/CallCenterData.xlsx` is small for deep learning (~86 effective training rows after windowing/splits), which is why LSTM underperforms MLP/CNN in the current metrics. Investigated NHS 111 (UK non-emergency medical helpline) open data as a real, larger, genuinely-healthcare-domain replacement (source: NHS England statistics, `20210708-NHS-111-MDS-time-series-to-March-2021.xlsx`). Two candidates were built and shown to the user, not yet integrated:
- **Candidate A** — national total calls/month, 128 months (Aug 2010–Mar 2021), single clean column, closest drop-in replacement for `Healthcare`. Caveat: first ~2.5 years are near-zero (national rollout, not real demand), which would distort the scaler's range — recommended trimming to ~2013 onward (~99 months) before use.
- **Candidate B** — 5 NHS 111 regional provider areas as parallel columns (mirrors the current file's 5-sector layout), 98 months (Feb 2013–Mar 2021).
- Recommendation given: trimmed Candidate A, because only one column is ever actually modeled — B's multi-column structure doesn't improve the thing that matters (the single modeled series), while A's regime-change confound is a simple trim.
- Repo was not a git repo as of this work (about to be initialized/pushed). `.venv/`, `__pycache__/`, `output/` contents, and `.claude/settings.local.json` are all gitignored. No secrets found in a scan.

## Working with this user

- Wants a direct recommendation/verdict when asking "which do you prefer" — not another round of options.
- Wants audits to actually execute and validate code (not just read it) before declaring something fixed.
