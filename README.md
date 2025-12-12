# Sales Forecast ML Package

A Python package for **reading**, **preprocessing**, **visualizing**, and **modeling** sales data. It provides utilities to:

- Load tabular sales data from CSV (`read.read_data`)
- Clean and engineer features (`preprocess.process_data`, `preprocess.prepare_data`)
- Visualize sales distributions by genre or platform (`viz.print_genre_distribution`, `viz.print_platform_distribution`)
- Train and tune a Random Forest model and generate predictions (`model.rf_fit`, `model.predict`)

---

## Overview
This package is designed to help you work with video game sales and activity data. It includes tools for reading raw CSV files, preprocessing and feature engineering, visualizing distributions, and building predictive models using Random Forest.

---

## Features
- **CSV Reader**: Reads a CSV file and applies basic type cleaning for year, drops an extraneous index column.
- **Data Processing**: Aggregates per `Name` and combines sales/metadata, removes duplicates.
- **Feature Preparation**: Multi-label binarization for `Platform` and `Genre`; concatenates with numeric predictors, drops missing `Year`.
- **Visualization**:
  - `print_genre_distribution(sales, genre, area)`: Histogram of sales for a given genre and region.
  - `print_platform_distribution(sales, platform, area)`: Histogram of sales for a given platform and region.
- **Modeling**: Grid search over Random Forest hyperparameters; evaluation prints R², RMSE (log scale), and top feature importances.
- **Prediction**: Scales inputs consistently and returns predictions on the original scale via `expm1`.

---

## Included Datasets
The package includes two sample datasets in the `data/` directory for quick experimentation:
- `data/vgsales.csv`: Raw sales data with columns like `Name`, `Platform`, `Year`, `Genre`, `Publisher`, regional sales, and `Global_Sales`.
- `data/game_data.csv`: Steam activity metrics (`all_time_peak`, `last_30_day_avg`) for integration with sales data aggrigated with the vgsales dataset.

Use `get_data_path(filename)` from `read.py` to access these files programmatically.

---

## Installation
```bash
pip install -U pip
pip install -U scikit-learn pandas numpy matplotlib seaborn
# If packaged for PyPI:
# pip install your-package-name
```

---

## Quickstart
```python
from your_package import (
    read_data, process_data, prepare_data,
    print_genre_distribution, print_platform_distribution,
    rf_fit, predict
)

# 1) Read
data = read_data("data/sales.csv")

# 2) Process & Prepare
sales_combined = process_data(data)
final_df = prepare_data(sales_combined)

# 3) Visualize
print_genre_distribution(data, genre="Action", area="Global_Sales")
print_platform_distribution(data, platform="PS4", area="Global_Sales")

# 4) Train model
best_model = rf_fit(final_df, area="Global_Sales")

# 5) Predict
new_data = final_df.drop(columns=["Global_Sales"]).iloc[:5]
preds = predict(best_model, area="Global_Sales", new_data=new_data)
print(preds)
```

---

## API Reference
### Top-Level Functions
- `read_data(file_path)`
- `process_data(df)`
- `prepare_data(df)`
- `print_genre_distribution(sales, genre, area)`
- `print_platform_distribution(sales, platform, area)`
- `rf_fit(final_df, area)`
- `predict(model, area, new_data)`

---

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Contributing
Issues and PRs are welcome. Please include reproducible examples and data schema.
