# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 07:19:21 2024

@author: admin
"""

import pandas as pd

# Load datasets
movies = pd.read_csv('C://Users//admin//Downloads//movies_metadata.csv')

# Display the first few rows
print(movies.head())
# Overview of the dataset
print(movies.info())

# Statistical summary
print(movies.describe())

# Checking for missing values
print(movies.isnull().sum())

# Assuming there's a 'country' column
# Adjust the column name based on your dataset
movies_india = movies[movies['country'].str.contains('India', case=False, na=False)].copy()

print(f"Total Indian Movies: {movies_india.shape[0]}")

# Display missing values
print(movies_india.isnull().sum())

# For simplicity, let's drop rows with missing target variable 'rating'
movies_india = movies_india.dropna(subset=['rating'])

# For other features, decide based on the context
# Example: Fill missing budget with median
movies_india['budget'] = pd.to_numeric(movies_india['budget'], errors='coerce')
movies_india['budget'] = movies_india['budget'].fillna(movies_india['budget'].median())

# Fill missing runtime with median
movies_india['runtime'] = pd.to_numeric(movies_india['runtime'], errors='coerce')
movies_india['runtime'] = movies_india['runtime'].fillna(movies_india['runtime'].median())

# For categorical features like 'director' or 'cast', fill missing with 'Unknown'
movies_india['director'] = movies_india['director'].fillna('Unknown')
movies_india['cast'] = movies_india['cast'].fillna('Unknown')
movies_india['genres'] = movies_india['genres'].fillna('Unknown')

# Select relevant columns
# Adjust column names based on your dataset
features = movies_india[['genres', 'director', 'cast', 'budget', 'runtime', 'release_date', 'rating']]

print(features.head())


import ast

def parse_list_column(column):
    """
    Parses a stringified list into a Python list.
    """
    try:
        return ast.literal_eval(column)
    except:
        return []

# Parse 'genres' if it's a stringified list of dictionaries
features['genres'] = features['genres'].apply(parse_list_column)
features['genres'] = features['genres'].apply(lambda x: [genre['name'] for genre in x] if isinstance(x, list) else x)

# Parse 'cast' assuming it's a stringified list of dictionaries
features['cast'] = features['cast'].apply(parse_list_column)
features['cast'] = features['cast'].apply(lambda x: [member['name'] for member in x[:3]] if isinstance(x, list) else x)  # Top 3 actors

# If 'director' is a stringified list or dictionary
# Adjust parsing based on your dataset's structure
features['director'] = features['director'].apply(parse_list_column)
features['director'] = features['director'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else 'Unknown')

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(features['genres']), columns=mlb.classes_, index=features.index)

# Drop the original 'genres' column and concatenate the encoded genres
features = features.drop('genres', axis=1)
features = pd.concat([features, genres_encoded], axis=1)

print(genres_encoded.head())


# Calculate mean rating for each director
director_mean = features.groupby('director')['rating'].mean()

# Map the mean to the director column
features['director_encoded'] = features['director'].map(director_mean)

# Handle directors with no ratings
features['director_encoded'] = features['director_encoded'].fillna(features['rating'].mean())

# Drop the original 'director' column
features = features.drop('director', axis=1)

from collections import Counter

# Flatten the list of all cast members
all_cast = Counter([actor for cast_list in features['cast'] for actor in cast_list if actor != 'Unknown'])

# Select top 50 most common actors
common_cast = [actor for actor, count in all_cast.most_common(50)]

# Create binary features for common actors
for actor in common_cast:
    features[f'cast_{actor}'] = features['cast'].apply(lambda x: 1 if actor in x else 0)

# Drop the original 'cast' column
features = features.drop('cast', axis=1)

print(features.head())

# Convert 'budget' and 'runtime' to numeric if not already
features['budget'] = pd.to_numeric(features['budget'], errors='coerce')
features['runtime'] = pd.to_numeric(features['runtime'], errors='coerce')

# Fill any remaining missing values with median
features['budget'] = features['budget'].fillna(features['budget'].median())
features['runtime'] = features['runtime'].fillna(features['runtime'].median())

# Convert 'release_date' to datetime and extract year
features['release_date'] = pd.to_datetime(features['release_date'], errors='coerce')
features['release_year'] = features['release_date'].dt.year

# Fill missing 'release_year' with median or a specific value
features['release_year'] = features['release_year'].fillna(features['release_year'].median())

# Drop the original 'release_date' column
features = features.drop('release_date', axis=1)

# Define target variable
y = features['rating']

# Define feature set by dropping the target
X = features.drop('rating', axis=1)

print(X.head())
print(y.head())

from sklearn.preprocessing import StandardScaler

# Identify numerical columns
numerical_cols = ['budget', 'runtime', 'release_year', 'director_encoded'] + [col for col in X.columns if col.startswith('cast_')]

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print(X.head())

from sklearn.model_selection import train_test_split

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)

# Predict on test set
y_pred_lr = lr.predict(X_test)

# Evaluate the model
print("Linear Regression:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_lr, squared=False):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}\n")


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Regressor:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_rf, squared=False):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}\n")


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Regressor:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_rf, squared=False):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}\n")

from sklearn.ensemble import GradientBoostingRegressor

# Initialize the model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gbr.fit(X_train, y_train)

# Predict on test set
y_pred_gbr = gbr.predict(X_test)

# Evaluate the model
print("Gradient Boosting Regressor:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_gbr):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_gbr, squared=False):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_gbr):.2f}\n")

import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(title)
    plt.show()

# Example for Random Forest
plot_predictions(y_test, y_pred_rf, 'Random Forest: Actual vs Predicted Ratings')

# Example for Gradient Boosting
plot_predictions(y_test, y_pred_gbr, 'Gradient Boosting: Actual vs Predicted Ratings')

# Calculate residuals
residuals_rf = y_test - y_pred_rf
residuals_gbr = y_test - y_pred_gbr

# Plot residual distribution for Random Forest
plt.figure(figsize=(8,6))
sns.histplot(residuals_rf, bins=50, kde=True, color='blue')
plt.xlabel('Residuals')
plt.title('Random Forest Residual Distribution')
plt.show()

# Plot residual distribution for Gradient Boosting
plt.figure(figsize=(8,6))
sns.histplot(residuals_gbr, bins=50, kde=True, color='green')
plt.xlabel('Residuals')
plt.title('Gradient Boosting Residual Distribution')
plt.show()


