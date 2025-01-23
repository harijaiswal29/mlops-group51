import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error ,r2_score

import warnings
warnings.filterwarnings("ignore")

# Part B: Load Data
train = pd.read_csv("data/house-prices-advanced-regression-techniques/train.csv")
test  = pd.read_csv("data/house-prices-advanced-regression-techniques/test.csv")
test_copy = test.copy()

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()


# Checking for missing values
missing_counts = train.isna().sum().sort_values(ascending=False)
missing_percent = (missing_counts / train.shape[0]) * 100
missing_df = pd.DataFrame({
    'Feature': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percent': missing_percent.values
})
print("Missing values (sorted by count):")
#display(missing_df[missing_df['Missing_Count']>0])


# Handling missing values in the training data (train)
# Replace missing categorical values with "None"
categorical_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                         'GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond', 
                         'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
train[categorical_fill_none] = train[categorical_fill_none].fillna("None")

# Replace missing numerical values with 0
numerical_fill_zero = ['GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                       'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars']
train[numerical_fill_zero] = train[numerical_fill_zero].fillna(0)

# Replace LotFrontage with median value by Neighborhood
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Fill Electrical with mode
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

# Replace missing values in GarageQual and GarageCond with "None"
train[['GarageQual', 'GarageCond']] = train[['GarageQual', 'GarageCond']].fillna("None")

# Repeat the same for the test dataset
# Replace missing categorical values with "None"
test[categorical_fill_none] = test[categorical_fill_none].fillna("None")

# Replace missing numerical values with 0
test[numerical_fill_zero] = test[numerical_fill_zero].fillna(0)

# Replace LotFrontage with median value by Neighborhood
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Fill Electrical with mode
if 'Electrical' in test.columns:
    test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])

# Replace missing values in GarageQual and GarageCond with "None"
if 'GarageQual' in test.columns and 'GarageCond' in test.columns:
    test[['GarageQual', 'GarageCond']] = test[['GarageQual', 'GarageCond']].fillna("None")


potential_outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)]
print(potential_outliers)

# Step 2: Calculate the median and standard deviation of SalePrice for Edwards neighborhood
edwards_stats = train[train['Neighborhood'] == 'Edwards']['SalePrice'].agg(['median', 'std'])
edwards_median = edwards_stats['median']
edwards_std = edwards_stats['std']

# Step 3: Check if the SalePrice deviates significantly from the neighborhood median
potential_outliers['PriceOutlier'] = (
    (potential_outliers['SalePrice'] < edwards_median - 2 * edwards_std) |
    (potential_outliers['SalePrice'] > edwards_median + 2 * edwards_std)
)

# Step 4: Output the results
print("Edwards Neighborhood Outliers Analysis:\n", 
      potential_outliers[['GrLivArea', 'SalePrice', 'PriceOutlier']])

# Step 5: Conclusion
if potential_outliers['PriceOutlier'].any():
    print("The points are statistical outliers in Edwards neighborhood.")
else:
    print("The points are not statistical outliers based on Edwards neighborhood characteristics.")

neighborhood_prices = train.groupby('Neighborhood')['SalePrice'].mean().sort_values()

# Correlation Analysis
train_cleaned = train.copy()

categorical_cols = train_cleaned.select_dtypes(include='object').columns
for col in categorical_cols:
    train_cleaned[col] = train_cleaned[col].astype(str)
    train_cleaned[col] = LabelEncoder().fit_transform(train_cleaned[col])

numeric_cols = train_cleaned.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

numeric_cols = train.select_dtypes(include=[np.number]).columns


# Select categorical/ordinal columns
selected_categorical_columns = [
    'Neighborhood', 'BldgType', 'HouseStyle', 
    'Exterior1st', 'KitchenQual', 'GarageType', 'SaleCondition'
]

# One-hot encode the selected categorical columns
train = pd.get_dummies(train, columns=selected_categorical_columns, drop_first=True)
test = pd.get_dummies(test, columns=selected_categorical_columns, drop_first=True)

# Fill any missing columns in the test set after alignment with zeros
test = test.fillna(0)

# Output the processed train and test datasets
train.head(), test.head()

train, test = train.align(test, join = 'left', axis = 1)


# -------------------------------------------------------------------
# Polynomial Feature: OverallQual^2

poly = PolynomialFeatures(degree=2, include_bias=False)
overallqual_2d = train['OverallQual'].values.reshape(-1, 1)
overallqual_poly = poly.fit_transform(overallqual_2d)

df_overallqual_poly = pd.DataFrame(
    overallqual_poly,
    columns=['OverallQual_poly1','OverallQual_Sq'],
    index=train.index
)
train = pd.concat([train, df_overallqual_poly], axis=1)

overallqual_2d_test = test['OverallQual'].values.reshape(-1, 1)
overallqual_poly_test = poly.transform(overallqual_2d_test)  # Use the same transformer as for train
df_overallqual_poly_test = pd.DataFrame(
    overallqual_poly_test,
    columns=['OverallQual_poly1', 'OverallQual_Sq'],
    index=test.index
)
test = pd.concat([test, df_overallqual_poly_test], axis=1)
# -------------------------------------------------------------------
# Indicator Variable: HasPool (1 if PoolArea > 0, else 0)
train['HasPool'] = (train['PoolArea'] > 0).astype(int)
test['HasPool'] = (test['PoolArea'] > 0).astype(int)
# -------------------------------------------------------------------
# Piecewise Features for GrLivArea
#    GrLivArea_Below1500 = min(GrLivArea, 1500)
#    GrLivArea_Above1500 = max(GrLivArea-1500, 0)
def piecewise_grlivarea(x):
    if x <= 1500:
        return x, 0
    else:
        return 1500, x - 1500

below_list = []
above_list = []
for idx, row in train.iterrows():
    below, above = piecewise_grlivarea(row['GrLivArea'])
    below_list.append(below)
    above_list.append(above)

train['GrLivArea_Below1500'] = below_list
train['GrLivArea_Above1500'] = above_list

# Piecewise Features for GrLivArea
below_list_test = []
above_list_test = []
for idx, row in test.iterrows():
    below, above = piecewise_grlivarea(row['GrLivArea'])
    below_list_test.append(below)
    above_list_test.append(above)

test['GrLivArea_Below1500'] = below_list_test
test['GrLivArea_Above1500'] = above_list_test

# -------------------------------------------------------------------
# Fill for newly introduced columns
for col in train.columns:
    if train[col].dtype in [np.float64, np.int64]:
        train[col].fillna(train[col].median(), inplace=True)
    else:
        train[col].fillna('None', inplace=True)


# CREATE NEW FEATURES: TotalBathrooms, HouseAge, RemodelAge, TotalSF

# 1. TotalBathrooms
bathroom_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
for col in bathroom_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    train[col].fillna(train[col].median(), inplace=True)

train['TotalBathrooms'] = (
    train['FullBath'] + 
    (train['HalfBath'] * 0.5) + 
    train['BsmtFullBath'] + 
    (train['BsmtHalfBath'] * 0.5)
)

for col in bathroom_cols:
    test[col] = pd.to_numeric(test[col], errors='coerce')
    test[col].fillna(train[col].median(), inplace=True)  # Use median from training set

test['TotalBathrooms'] = (
    test['FullBath'] + 
    (test['HalfBath'] * 0.5) + 
    test['BsmtFullBath'] + 
    (test['BsmtHalfBath'] * 0.5)
)
# 2. HouseAge
train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge'] = test['YrSold'] - test['YearBuilt']

# 3. RemodelAge
train['RemodelAge'] = train['YrSold'] - train['YearRemodAdd']
test['RemodelAge'] = test['YrSold'] - test['YearRemodAdd']

# 4. TotalSF
sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
for col in sf_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    train[col].fillna(train[col].median(), inplace=True)

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

for col in sf_cols:
    test[col] = pd.to_numeric(test[col], errors='coerce')
    test[col].fillna(train[col].median(), inplace=True)  # Use median from training set

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']


# FILL FOR NEWLY INTRODUCED COLUMNS

for col in train.columns:
    if train[col].dtype in [np.float64, np.int64]:
        train[col].fillna(train[col].median(), inplace=True)
    else:
        train[col].fillna('None', inplace=True)


#Linear Regression Model
numerical_features = ['GrLivArea', 'LotArea', 'OverallQual', 'YearBuilt']
selected_categorical_columns = ['Neighborhood', 'BldgType', 'HouseStyle', 'Exterior1st', 'KitchenQual', 'GarageType', 'SaleCondition']
new_feature_cols = [
    'OverallQual_Sq',       # polynomial
    'HasPool',              # indicator
    'GrLivArea_Below1500',  # piecewise
    'GrLivArea_Above1500',  # piecewise
    'TotalBathrooms',       # new feature
    'HouseAge',             # new feature
    'RemodelAge',           # new feature
    'TotalSF'               # new feature
]

# Combine numerical features and one-hot encoded categorical columns
encoded_categorical_columns = [col for col in train.columns if col not in train.columns or col in selected_categorical_columns]
selected_features = numerical_features + encoded_categorical_columns + new_feature_cols
test = test[selected_features]

# Separate features (X) and target (y)
X = train[selected_features] 
y = train['SalePrice']

# Log-transform the target variable
#y = np.log1p(y)  # log(SalePrice + 1)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.2, 
                                                  random_state=42)

# Fit a Linear Regression model
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)

# Evaluate
preds_val = linreg_model.predict(X_val)
rmse_val = np.sqrt(np.mean((preds_val - y_val)**2))
r2_val = r2_score(y_val, preds_val)

# Display results
print("Validation RMSE:", rmse_val)
print("Validation R-squared:", r2_val)


# Ridge model
ridge_model = Ridge(alpha=10)

#Fit the Ridge model
ridge_model.fit(X_train, y_train)

#Predict on the validation set
ridge_preds = ridge_model.predict(X_val)

#Evaluate the model (RMSE)
ridge_mse = mean_squared_error(y_val, ridge_preds)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_val, ridge_preds)

# Display results
print("Ridge Regression RMSE:", ridge_rmse)
print("Ridge Regression R-squared:", ridge_r2)



# 5-Fold Cross-Validation for Linear Regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lin = cross_val_score(
    linreg_model, X, y, 
    scoring='neg_root_mean_squared_error',
    cv=kf
)
cv_rmse_lin = -np.mean(cv_scores_lin)

cv_r2_scores_lin = cross_val_score(
    linreg_model, X, y, 
    scoring='r2',
    cv=kf
)
cv_r2_lin = np.mean(cv_r2_scores_lin)

print("Linear Regression 5-Fold CV RMSE:", cv_rmse_lin)
print("Linear Regression 5-Fold CV R²:", cv_r2_lin)


# 5-Fold Cross-Validation for Ridge Regression
cv_scores_ridge = cross_val_score(
    ridge_model, X, y, 
    scoring='neg_root_mean_squared_error',
    cv=kf
)
cv_rmse_ridge = -np.mean(cv_scores_ridge)

cv_r2_scores_ridge = cross_val_score(
    ridge_model, X, y, 
    scoring='r2',
    cv=kf
)
cv_r2_ridge = np.mean(cv_r2_scores_ridge)

print("Ridge Regression 5-Fold CV RMSE:", cv_rmse_ridge)
print("Ridge Regression 5-Fold CV R²:", cv_r2_ridge)


# Predictions for test set
test_preds = ridge_model.predict(test)

# Save predictions to CSV
submission = pd.DataFrame({
    'Id': test_copy['Id'],
    'SalePrice': test_preds
})

print(submission)