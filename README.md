# Deep Learning Call Center Forecasting

Deep Learning has become a fundamental part of the new generation of Time Series Forecasting models, obtaining excellent results.
While in classical Machine Learning models — such as autoregressive models (AR) or exponential smoothing — feature engineering is performed manually and often some parameters are optimized also considering the domain knowledge, Deep Learning models learn features and dynamics only and directly from the data.
Thanks to this, they speed up the process of data preparation and are able to learn more complex data patterns in a more complete way.

This project forecasts monthly call center volume (Healthcare sector) using four architectures: MLP, CNN, LSTM, and CNN-LSTM.

## Project structure

```
.
├── data/                    # Input dataset
│   └── CallCenterData.xlsx
├── notebooks/               # Exploratory notebook (same flow as src/, for Colab/Jupyter use)
│   └── DeepLearning.ipynb
├── docs/                    # Supplementary reference material
│   ├── architecture_diagram.png
│   └── cnnrnn_reference.pptx
├── src/                     # Modular pipeline
│   ├── engine.py            # Entry point: loads data, trains & evaluates all 4 models
│   └── ml_pipeline/
│       ├── utils.py          # Shared paths, reshape helper, metrics, plotting
│       ├── mlp.py
│       ├── cnn_model.py
│       ├── lstm_model.py
│       └── cnn_lstm_model.py
├── output/                  # Generated at runtime (gitignored)
│   ├── models/               # Trained models, native Keras format (.keras)
│   ├── plots/                 # Visualization + prediction plots
│   └── model_comparison.csv   # MAE/RMSE/MAPE for all 4 models, sorted by RMSE
└── requirements.txt
```

## Models

| Model | Architecture |
|---|---|
| MLP | `Dense(100, relu) -> Dense(1)` |
| CNN | `Conv1D(64, k=2) -> MaxPool1D -> Flatten -> Dense(50) -> Dense(1)` |
| LSTM | `LSTM(50, relu) -> Dense(1)` |
| CNN-LSTM | `TimeDistributed(Conv1D(64, k=1)) -> TimeDistributed(MaxPool1D) -> TimeDistributed(Flatten) -> LSTM(50) -> Dense(1)` |

Univariate time series are a dataset comprised of a single series of observations with a temporal ordering, and a model is required to learn from the series of past observations to predict the next value in the sequence. Before a univariate series can be modeled, it must be prepared: the sequence of observations is transformed into multiple (past-window -> next-value) training examples.

### Correlation vs. AutoCorrelation

- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. The correlation coefficient varies between +1 and -1; a value of +-1 indicates a perfect degree of association, and as it approaches 0 the relationship weakens.
- Auto-correlation refers to the case when your errors are correlated with each other — if the current observation of your dependent variable is correlated with past observations, you're in the trap of auto-correlation.

### Time series basics

- Chronological data — cannot be shuffled
- Each row represents a specific time record
- Train/test splits happen chronologically
- Data is analyzed univariately (for this use case)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
```

## Running the pipeline

```bash
cd src
python engine.py
```

This will:
1. Load `data/CallCenterData.xlsx` and plot each sector's time series to `output/plots/`.
2. Build lagged training windows from the Healthcare series.
3. Split into train/validation/test, all chronologically (most recent 20 points as test, most recent ~10% of the remainder as validation) — scaler fit on train only, to avoid leakage.
4. Train MLP, CNN, LSTM, and CNN-LSTM, saving prediction plots (actual vs. predicted, with legend) to `output/plots/` and trained models in native Keras format to `output/models/`.
5. Print and save a MAE/RMSE/MAPE comparison across all 4 models to `output/model_comparison.csv`.

All Python/NumPy/TensorFlow RNGs are seeded at the start of the run, so results are identical across runs on the same machine.

Alternatively, open `notebooks/DeepLearning.ipynb` in Jupyter or Google Colab to run the same flow interactively — it mirrors `src/` exactly (same splits, architectures, seeding, and metrics), just self-contained in one notebook rather than split into modules.
