def main():
    print("Hello from stat-386-final-project!")


if __name__ == "__main__":
    main()

def read_data(file_path):
    import pandas as pd
    """Reads data from a given file path as a csv."""
    sales = pd.read_csv(file_path)
    sales['Year'] = sales['Year'].astype('Int64')
    sales.drop(columns=['Unnamed: 0'], inplace=True)
    return sales

def process_data(sales):
    import pandas as pd
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

def print_genre_distribution(sales, genre, area):
    import seaborn as sns
    import matplotlib.pyplot as plt 
    """Prints the distribution of sales in area for genre."""
    sns.histplot(data=sales[sales['Genre'].astype(str).str.contains(genre, na=False)], x=area, bins=50)
    plt.show()

def print_platform_distribution(sales, platform, area):
    import seaborn as sns
    import matplotlib.pyplot as plt 
    """Prints the distribution of sales in area for platform."""
    sns.histplot(data=sales[sales['Platform'].astype(str).str.contains(platform, na=False)], x=area, bins=50)
    plt.show()

def prepare_data(sales_combined):
    from sklearn.preprocessing import MultiLabelBinarizer
    import pandas as pd
    sales_combined['Platform'] = sales_combined['Platform'].apply(lambda x: x if isinstance(x, list) else [])
    sales_combined['Genre'] = sales_combined['Genre'].apply(lambda x: x if isinstance(x, list) else [])

    mlb_platform = MultiLabelBinarizer()
    mlb_genre = MultiLabelBinarizer()


    # Apply encoding
    platform_encoded = pd.DataFrame(mlb_platform.fit_transform(sales_combined['Platform']), columns=[f"Platform_{cat}" for cat in mlb_platform.classes_])
    genre_encoded = pd.DataFrame(mlb_genre.fit_transform(sales_combined['Genre']), columns=[f"Genre_{cat}" for cat in mlb_genre.classes_])


    # Combine encoded columns with numeric predictors
    final_df = pd.concat([sales_combined[['all_time_peak', 'last_30_day_avg', 'Year', 'Global_Sales']], platform_encoded, genre_encoded], axis=1)
    final_df = final_df.dropna(subset=['Year'])
    return final_df

def rf_fit(final_df, area):
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pandas as pd

    x = final_df.drop(columns = [area])
    y = np.log1p(final_df[area])

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import GridSearchCV

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train[['all_time_peak', 'last_30_day_avg', 'Year']] = scaler.fit_transform(X_train[['all_time_peak', 'last_30_day_avg', 'Year']])
    X_test[['all_time_peak', 'last_30_day_avg', 'Year']] = scaler.transform(X_test[['all_time_peak', 'last_30_day_avg', 'Year']])

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [1, 3, 5],
        'min_samples_leaf': [1, 2, 3]
    }

    # GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions and evaluation
    preds = best_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds)

    print("Best Parameters:", grid_search.best_params_)
    print("R²:", r2)
    print("RMSE (log scale):", rmse)
    print("Top 10 Feature Importances:")
    importances = pd.Series(best_model.feature_importances_, index=x.columns).sort_values(ascending=False)
    print(importances.head(10))

    return

    


