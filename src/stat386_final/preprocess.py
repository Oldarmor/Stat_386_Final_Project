import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def process_data(sales):
    """Processes the sales data and drop duplicates."""
    sales = sales.drop_duplicates()
    sales_combined = (
        sales.groupby('Name')
        .agg({
            'Platform': lambda x: list(pd.unique(x.dropna())),
            'Year': 'max',
            'Genre': lambda x: list(pd.unique(x.dropna())),
            'Publisher': lambda x: list(pd.unique(x.dropna())),
            'all_time_peak': 'max',
            'last_30_day_avg': 'max',
            'NA_Sales': 'sum',
            'EU_Sales': 'sum',
            'JP_Sales': 'sum',
            'Other_Sales': 'sum',
            'Global_Sales': 'sum',
        })
        .reset_index()
    )
    return sales_combined

def prepare_data(sales_combined):
    """Completes data preparation."""
    sales_combined['Platform'] = sales_combined['Platform'].apply(lambda x: x if isinstance(x, list) else [])
    sales_combined['Genre'] = sales_combined['Genre'].apply(lambda x: x if isinstance(x, list) else [])

    mlb_platform = MultiLabelBinarizer()
    mlb_genre = MultiLabelBinarizer()


    # Apply encoding
    platform_encoded = pd.DataFrame(mlb_platform.fit_transform(sales_combined['Platform']), columns=[f"Platform_{cat}" for cat in mlb_platform.classes_])
    genre_encoded = pd.DataFrame(mlb_genre.fit_transform(sales_combined['Genre']), columns=[f"Genre_{cat}" for cat in mlb_genre.classes_])


    # Combine encoded columns with numeric predictors
    final_df = pd.concat([sales_combined[['all_time_peak', 'last_30_day_avg', 'Year', 'Rank', 'Global_Sales']], platform_encoded, genre_encoded], axis=1)
    final_df = final_df.dropna(subset=['Year'])
    return final_df