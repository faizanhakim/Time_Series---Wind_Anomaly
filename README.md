# TS Anomaly Detection (Wind Turbine SCADA)

## Overview
PyTorch LSTM-based anomaly detection pipeline for wind turbine SCADA data. Preprocesses data into sequences, trains an LSTM to reconstruct sequences, then flags sequences with high reconstruction error.

## Setup
1. Create venv:
   python -m venv venv
   source venv/bin/activate
2. Install:
   pip install -r requirements.txt

## Data (Kaggle)
- Download a wind turbine SCADA dataset (e.g., "Turbine SCADA" or "Wind Turbine SCADA Dataset") from Kaggle.
- Place CSV file(s) into `data/` and ensure the timestamp column is named `timestamp` (or edit loader).

## Pipeline
1. Preprocess:
   python ts/data_prep.py -- (or import preprocess_for_training)
   Example:
   python -c "from ts.data_prep import preprocess_for_training; preprocess_for_training('data/your_scada.csv')"

2. Train:
   python ts/train.py --data_dir data/processed --out_dir models --epochs 30

3. Evaluate:
   python ts/evaluate.py --model models/model_final.pt --test data/processed/test.npy

## Notes
- The repo provides a reproducible pipeline and modular scripts for extension.
- Add labeled anomalies to compute precision/recall if available.

