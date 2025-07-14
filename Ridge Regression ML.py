# ridge_regression_assignment_script.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. Load data
ridge_data = pd.read_csv('ridge_reg_data.csv')

# 2. Split data into features (X) and target (y)
X = ridge_data.drop(columns='target')
y = ridge_data['target']

# 3. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Ordinary Least Squares Linear Regression
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# 5. Stochastic Gradient Descent Regressor
sgd_model = SGDRegressor(
    max_iter=10000,
    penalty=None,
    learning_rate='constant',
    n_iter_no_change=100,
    random_state=42
)
sgd_model.fit(X_train, y_train)

# 6. Ridge Regression with Grid Search
ridge_model = Ridge(random_state=42)
ridge_param_grid = {
    'alpha': [0.25, 0.50, 0.75, 1, 2, 3, 5, 10, 100]
}
grid_search_cv_ridge = GridSearchCV(
    ridge_model, 
    ridge_param_grid, 
    verbose=1, 
    cv=10
)
grid_search_cv_ridge.fit(X_train, y_train)
ridge_model = grid_search_cv_ridge.best_estimator_

# 7. Make predictions
ols_pred = ols_model.predict(X_test)
sgd_pred = sgd_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# 8. Calculate RMSE (rounded to 4 decimals)
rmse_ols = round(root_mean_squared_error(y_test, ols_pred), 4)
rmse_sgd = round(root_mean_squared_error(y_test, sgd_pred), 4)
rmse_ridge = round(root_mean_squared_error(y_test, ridge_pred), 4)

# 9. Calculate R^2 (rounded to 4 decimals)
r2_ols = round(r2_score(y_test, ols_pred), 4)
r2_sgd = round(r2_score(y_test, sgd_pred), 4)
r2_ridge = round(r2_score(y_test, ridge_pred), 4)

# 10. Print results
print(f"RMSE (OLS): {rmse_ols}")
print(f"RMSE (SGD): {rmse_sgd}")
print(f"RMSE (Ridge): {rmse_ridge}")
print("-----------")
print(f"R-squared (OLS): {r2_ols}")
print(f"R-squared (SGD): {r2_sgd}")
print(f"R-squared (Ridge): {r2_ridge}")
