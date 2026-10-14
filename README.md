# 🚗 Car Price Prediction Using Machine Learning

A Python machine learning project that estimates the selling price of a used car from vehicle specifications. The project includes exploratory data analysis (EDA), data preprocessing, an **XGBoost regression model**, model evaluation, and saved model artifacts for inference.


## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Installation](#installation)
- [Run the Prediction Script](#run-the-prediction-script)
- [Model Training Workflow](#model-training-workflow)
- [Evaluation Metrics](#evaluation-metrics)
- [Saved Model Files](#saved-model-files)
- [Important Notes](#important-notes)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

The project uses historical used-car data to learn the relationship between vehicle characteristics and selling price. The Jupyter Notebook performs data inspection, statistical analysis, visualizations, preprocessing, model training, and evaluation.

The notebook uses `XGBRegressor` from XGBoost and saves the trained model and feature-column information with Joblib. The repository also includes an `app.py` inference script intended to load the saved artifacts and predict a price for supplied car details.

## Objectives

- Explore used-car data using Pandas and visualizations.
- Identify and inspect missing values and data distributions.
- Analyze selling price against vehicle characteristics.
- Train a regression model to estimate used-car prices.
- Evaluate predictions with regression metrics.
- Save the model and feature columns for later inference.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, cleaning, and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Train/test splitting, grid search, and evaluation metrics |
| XGBoost | Regression model |
| Joblib | Saving and loading model artifacts |
| Jupyter Notebook | EDA and model development |

## Project Structure

```text
Car-Price-Prediction/
├── README.md
├── app.py
├── Car_Price_model_Predicts.ipynb
├── output_data.csv
├── datasets/
│   ├── cardekho.csv
│   ├── Clean Data_pakwheels.csv
│   └── used_cars.csv
└── model/
    ├── car_price_model.pkl
    ├── feature_columns.pkl
    └── model_name.pkl
```

**Files and directories**

- `app.py` — Prediction/inference script.
- `Car_Price_model_Predicts.ipynb` — Notebook for data analysis, feature preparation, model training, evaluation, and saving artifacts.
- `output_data.csv` — Optional output or example data file; its exact contents depend on the repository version.
- `datasets/` — Raw or prepared CSV datasets. The notebook currently loads `datasets/cardekho.csv`.
- `model/` — Serialized model and supporting files used for inference.

## Dataset

The notebook loads the CarDekho dataset from:

```text
datasets/cardekho.csv
```

The notebook analyzes columns including `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage(km/ltr/kg)`, `engine`, `max_power`, and `seats`.

The target column used in the model-training section is:

```python
TARGET_COLUMN = "selling_price"
```

Other CSV files are present in the described repository structure, but the supplied notebook's initial data-loading cell uses `cardekho.csv`. Do not combine datasets without checking their schemas, units, currency, and duplicate records first.

## Features Used

The notebook's exploratory analysis examines the following variables:

| Feature | Description |
|---|---|
| `name` | Car name/model, if retained in the prepared data |
| `year` | Year of manufacture |
| `km_driven` | Distance driven in kilometers |
| `fuel` | Fuel category |
| `seller_type` | Seller category |
| `transmission` | Transmission type |
| `owner` | Ownership category |
| `mileage(km/ltr/kg)` | Mileage value and unit as represented in the dataset |
| `engine` | Engine displacement as represented in the dataset |
| `max_power` | Maximum power as represented in the dataset |
| `seats` | Number of seats |
| `selling_price` | Target value to predict |

The final model input columns are determined by `FEATURE_COLUMNS` in the notebook. The exact columns depend on the `final_df` preprocessing steps, so the inference code must use the same feature names, transformations, and column order.

## Installation

### Prerequisites

- Python installed on your computer.
- `pip` package manager.
- The project datasets and model artifacts in their expected directories.

### 1. Open the project directory

```bash
cd Car-Price-Prediction
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

For notebook analysis and training:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

If you only run `app.py`, install the packages that the script imports and the dependencies required to load the saved model. The exact inference requirements depend on the implementation of `app.py`.

## Run the Prediction Script

Make sure the model files exist in the `model/` directory, then run:

```bash
python app.py
```

The repository description indicates that `app.py` defines a `predict_car_price()` function and includes an example call. The function's accepted arguments and output formatting should match the actual implementation in `app.py`.

Example import pattern:

```python
from app import predict_car_price

# Supply values using the exact arguments expected by app.py.
# Example:
# price = predict_car_price(...)
# print(price)
```

A specific prediction value is not included here because output depends on the input values and the saved model.

## Model Training Workflow

Open the training notebook:

```bash
jupyter notebook
```

Then open `Car_Price_model_Predicts.ipynb` and run the cells in order.

The notebook's workflow includes:

1. **Load data:** Read `datasets/cardekho.csv` with Pandas.
2. **Inspect data:** Check dimensions, column names, data types, summary statistics, and missing values.
3. **Handle missing values:** Apply the notebook's missing-value handling steps.
4. **Explore the data:** Calculate group statistics and examine relationships between price and vehicle characteristics.
5. **Visualize patterns:** Create correlation heatmaps, histograms, box plots, bar charts, and scatter plots.
6. **Prepare features:** Construct the final feature table used for training.
7. **Split data:** Create training and test subsets with `train_test_split`.
8. **Tune and train:** Use `GridSearchCV` with an `XGBRegressor` and train a final regression model.
9. **Evaluate:** Calculate RMSE, MSE, MAE, and R² on the test set.
10. **Save artifacts:** Store the trained model, feature columns, and model identifier under `model/`.

### Retrain and save the model

The notebook contains these save operations:

```python
joblib.dump(final_model, "model/car_price_model.pkl")
joblib.dump(FEATURE_COLUMNS, "model/feature_columns.pkl")
joblib.dump("Car_Price_Preict", "model/model_name.pkl")
```

Run the notebook from the project root, and ensure the `model/` directory exists before saving. After retraining, test the inference script to confirm it uses the newly saved artifacts and compatible preprocessing.

## Evaluation Metrics

The notebook evaluates the trained model using:

| Metric | Meaning |
|---|---|
| **MAE** — Mean Absolute Error | Average absolute difference between actual and predicted prices |
| **MSE** — Mean Squared Error | Average squared prediction error; larger errors receive greater weight |
| **RMSE** — Root Mean Squared Error | Square root of MSE, expressed in the target's units |
| **R² Score** | Indicates how well predictions explain target variation relative to a baseline |

The notebook prints these metrics during evaluation. Add the actual measured scores to this README after running the notebook; scores are not stated here because they depend on the current dataset and execution.

## Saved Model Files

| File | Purpose |
|---|---|
| `car_price_model.pkl` | Trained XGBoost model |
| `feature_columns.pkl` | Feature-column names/order used for model input |
| `model_name.pkl` | Saved model identifier |

**Security:** Only load `.pkl` files from sources you trust. Deserializing untrusted pickle/Joblib artifacts can execute malicious code.

## Important Notes

- **Feature consistency:** Training and inference must use the same features, category handling, units, and feature order.
- **Dataset quality:** Missing values, outliers, duplicate rows, and inconsistent data can affect prediction quality.
- **Model tuning:** Review the parameter names in the notebook's `GridSearchCV` grid. XGBoost parameter names must match the installed XGBoost version; misspelled or unused parameters may prevent effective tuning.
- **Validation:** Hyperparameter tuning should be performed on training data, with final performance reported on a separate held-out test set.
- **Market context:** Actual resale prices vary with location, condition, service history, demand, and market timing. The listed features may not capture all of these factors.
- **Artifact paths:** Run commands from the project root unless the code explicitly handles paths relative to its own file location.

## Future Improvements

- Package preprocessing and the estimator in a single Scikit-learn-compatible pipeline.
- Correct and validate the hyperparameter grid, then tune using training data only.
- Compare XGBoost with baseline models and other regression algorithms.
- Add cross-validation and residual/error analysis.
- Validate prediction inputs and handle unknown categories safely.
- Add automated tests for the prediction function.
- Create a web interface or REST API for easier use.
- Document the dataset source and its usage license.

## License

No license terms were provided with the project materials. If you plan to publish or redistribute this repository, add an appropriate `LICENSE` file and check the usage terms for each dataset. Retain required attribution to the original author and dataset providers.

---

**Built with Python, Pandas, Scikit-learn, and XGBoost.**
