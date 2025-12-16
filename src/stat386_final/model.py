import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def rf_fit(final_df, area):
    """Fits Random Forest model and returns the best model."""
    cols_drop = [col for col in final_df.columns if 'Sales' in col]
    cols_drop.append('Rank')
    x = final_df.drop(columns = cols_drop)
    y = np.log1p(final_df[area])



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
        'min_samples_split': [2, 3, 5],
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

    print(f'For Area: {area}')
    print("Best Parameters:", grid_search.best_params_)
    print("R²:", r2)
    print("RMSE (log scale):", rmse)
    print("Top 10 Feature Importances:")
    importances = pd.Series(best_model.feature_importances_, index=x.columns).sort_values(ascending=False)
    print(importances.head(10))

    return best_model, scaler

def predict(best_model, area, new_data, scaler=None):
    """Make predictions with a fitted model.

    If `scaler` is None, a `StandardScaler()` instance will be created (this
    allows tests to patch `StandardScaler` and provide a pre-fitted scaler).
    """
    # Preprocess new data
    new_data_scaled = new_data.copy()

    if scaler is None:
        scaler = StandardScaler()

    # Scale the numeric columns if they exist in the input data.
    cols_to_scale = ['all_time_peak', 'last_30_day_avg', 'Year', 'Rank']
    cols_present = [c for c in cols_to_scale if c in new_data_scaled.columns]
    if cols_present:
        new_data_scaled[cols_present] = scaler.transform(new_data_scaled[cols_present])

    # Predict and inverse-transform the log scale
    preds = best_model.predict(new_data_scaled)
    preds_exp = np.expm1(preds)  # Inverse of log1p

    return preds_exp