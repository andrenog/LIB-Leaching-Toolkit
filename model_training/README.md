# Model Training

This folder contains scripts and utilities for training machine learning models to predict properties of metal complexes based on experimental and molecular data.

## Main Script

- **train_models.py**: Main script for training various regression models (Random Forest, Gradient Boosting, MLP, and their pairwise-difference variants) on input data. It supports hyperparameter optimization with Optuna, feature engineering, and exports results and trained models.

## Features

- Reads input data from Excel files.
- Calculates molecular descriptors using RDKit.
- Supports multiple regression models and pairwise-difference approaches.
- Hyperparameter optimization via Optuna.
- Saves trained models and predictions to the `out/` directory with unique run identifiers.
- Outputs training and test statistics to Excel.

## Requirements

- **Python 3.12** is required.

Install dependencies with:
```bash
pip install -r requirements_training.txt
```

## Virtual Environment (Recommended)

It is recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements_training.txt
```

## Usage

1. Place your input Excel file (e.g., `input_913pts_250407.xlsx`) in the `data/` directory.
2. Adjust the `dataPath` variable in `train_models.py` if needed.
3. Run the script:
   ```bash
   python train_models.py
   ```
4. Outputs (models, predictions, logs, and statistics) will be saved in a new subfolder under `out/` for each run.

## Customization

- To enable or disable specific models or feature sets, edit the `model_configurations` list in `train_models.py`.
- To make predictions on new data, use the provided (commented) case study prediction section at the end of the script.

## Notes

- Some models (e.g., MLP) require missing values to be filled; the script handles this automatically.
- The script is designed for batch experimentation and reproducibility.

---
