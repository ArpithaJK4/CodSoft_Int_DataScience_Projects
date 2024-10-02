import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score




# Use double backslashes to escape the backslashes in the file path
#dataset = pd.read_csv("F:\Titanic-Dataset.csv")
dataset = pd.read_csv("F:\\Titanic-Dataset.csv")

# Print the dataset to check the contents
#print(dataset)
#dataset.head()
print(dataset.head())

print(dataset.shape)
print(dataset.info())
#to check no of missing value in each column
#print(dataset.isnull().sum())

#handling the missing value
#drop cabin column
dataset=dataset.drop(columns='Cabin', axis=1)

# replacing the missing values in'age' column with mean value 
#dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

print(dataset.isnull().sum())

#finding the mode value for embarked column
# mode means most repeating
print(dataset['Embarked'].mode())
print(dataset['Embarked'].mode()[0])
#replacing missing value in embarked column
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
print(dataset.isnull().sum())

#getting the statestical mesure about the data
print(dataset.describe())

#Finding the no odf people Survived and not servived

print(dataset['Survived'].value_counts()) 

##
#####Data Visualization


print(sns.set())
