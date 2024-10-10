# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 05:16:50 2024

@author:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
data_titanic = pd.read_csv("F:/Titanic-Dataset.csv")
print(data_titanic.head())
print(data_titanic.shape)
print(data_titanic.info())
print(data_titanic.isnull().sum())

# Data cleaning
# Drop unnecessary columns
data_titanic = data_titanic.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)

#
data_titanic['Age'] = data_titanic['Age'].fillna(data_titanic['Age'].median())
data_titanic['Embarked'] = data_titanic['Embarked'].fillna(data_titanic['Embarked'].mode()[0])
data_titanic = pd.get_dummies(data_titanic, columns=['Sex', 'Embarked'], drop_first=True)


# Verify no missing values remain
print(data_titanic.isnull().sum())

# Exploratory Data Analysis
sns.set(style="whitegrid")

plt.figure(figsize=(10,6))
sns.countplot(x='Survived', data=data_titanic)
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x='Sex', hue='Survived', data=data_titanic)
plt.title('Survival by Sex')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data_titanic['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data_titanic['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(15, 9))
sns.heatmap(data_titanic.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Encoding categorical variables using One-Hot Encoding
data_titanic = pd.get_dummies(data_titanic, columns=['Sex', 'Embarked'], drop_first=True)

# Feature and target separation
X = data_titanic.drop(columns=['Survived'], axis=1)
Y = data_titanic['Survived']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Pipeline creation with StandardScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', solver='liblinear'))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, Y_train)

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Predictions
Y_train_pred = best_model.predict(X_train)
Y_test_pred = best_model.predict(X_test)

# Evaluation
print('Training Accuracy:', accuracy_score(Y_train, Y_train_pred))
print('Test Accuracy:', accuracy_score(Y_test, Y_test_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_test_pred))
print('Classification Report:\n', classification_report(Y_test, Y_test_pred))

# Cross-Validation Scores
cv_scores = cross_val_score(best_model, X_train, Y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
