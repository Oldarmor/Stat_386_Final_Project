"""
stat386_final: utilities for reading, processing, visualizing, and modeling sales data.

Top-level API:
- read_data(file_path)
- process_data(df)
- prepare_data(df)
- print_genre_distribution(sales, genre, area)
- print_platform_distribution(sales, platform, area)
- rf_fit(final_df, area)
- predict(model, scaler, area, new_data)
"""

# Semantic version (update as you release)
__version__ = "0.1.5"

# Import and re-export the public API
from .read import read_data
from .preprocess import process_data, prepare_data
from .viz import print_genre_distribution, print_platform_distribution
from .model import rf_fit, predict

# Explicitly define the import surface
__all__ = [
    "read_data",
    "process_data",
    "prepare_data",
    "print_genre_distribution",
    "print_platform_distribution",
    "rf_fit",
    "predict",
    "__version__",
]
