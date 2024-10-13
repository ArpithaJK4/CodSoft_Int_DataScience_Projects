# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:00:11 2024

@author: admin
"""

import pandas as pd

# Load the dataset
# Replace 'creditcard.csv' with the path to your dataset if necessary
data = pd.read_csv('creditcard.csv')

# Display the first few rows
print(data.head())
# Overview of the dataset
print(data.info())

# Statistical summary
print(data.describe())

# Checking for missing values
print(data.isnull().sum())
import seaborn as sns
import matplotlib.pyplot as plt

# Countplot for Class distribution
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Percentage distribution
print(data['Class'].value_counts(normalize=True) * 100)

# Checking for duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Removing duplicates if any
data = data.drop_duplicates()
print(f"Data shape after removing duplicates: {data.shape}")


from imblearn.over_sampling import SMOTE

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print(y_resampled.value_counts())

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the 'Amount' and 'Time' features
X_resampled[['Time', 'Amount']] = scaler.fit_transform(X_resampled[['Time', 'Amount']])

# Verify scaling
print(X_resampled[['Time', 'Amount']].head())

from sklearn.model_selection import train_test_split

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    stratify=y_resampled)

# Check the shape of the splits
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts()}")
print(f"y_test distribution:\n{y_test.value_counts()}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the model
lr = LogisticRegression(solver='liblinear')  # 'liblinear' is suitable for small datasets

# Train the model
lr.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = lr.predict(X_test)

# Evaluation
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


from sklearn.metrics import ConfusionMatrixDisplay

# For Logistic Regression
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# For Random Forest
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title('Random Forest Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

# Already printed above for both models

from sklearn.metrics import roc_auc_score, roc_curve

# Logistic Regression
y_prob_lr = lr.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_prob_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

# Random Forest
y_prob_rf = rf.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Grid Search
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                              param_grid_rf,
                              cv=3,
                              n_jobs=-1,
                              scoring='f1',  # F1-score is useful for imbalanced classes
                              verbose=2)

# Fit Grid Search
grid_search_rf.fit(X_train, y_train)

# Best parameters
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")

# Best estimator
best_rf = grid_search_rf.best_estimator_

# Predict on test set
y_pred_best_rf = best_rf.predict(X_test)

# Evaluation
print("Tuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best_rf))


from imblearn.over_sampling import ADASYN

# Initialize ADASYN
adasyn = ADASYN(random_state=42)

# Apply ADASYN
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

# Check new class distribution
print(y_adasyn.value_counts())

from imblearn.under_sampling import RandomUnderSampler

# Initialize Random Under-Sampler
rus = RandomUnderSampler(random_state=42)

# Apply Random Under-Sampling
X_rus, y_rus = rus.fit_resample(X, y)

# Check new class distribution
print(y_rus.value_counts())

