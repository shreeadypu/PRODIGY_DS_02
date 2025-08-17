# Titanic Data Cleaning & EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# ---------------- Data Cleaning ---------------- #

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Verify cleaning
print("\nMissing values after cleaning:\n", df.isnull().sum())

# ---------------- Exploratory Data Analysis ---------------- #

# 1. Survival distribution
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title("Survival Count")
plt.show()

# 2. Gender vs Survival
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title("Survival by Gender")
plt.show()

# 3. Pclass vs Survival
sns.countplot(x='Pclass', hue='Survived', data=df, palette='muted')
plt.title("Survival by Passenger Class")
plt.show()

# 4. Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title("Age Distribution of Passengers")
plt.show()

# 5. Age vs Survival
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Age', data=df, palette='coolwarm')
plt.title("Age vs Survival")
plt.show()

# 6. Correlation Heatmap
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()


# ---------------- Key Insights ---------------- #
print("\nKey Insights:")
print("- Females had a much higher survival rate compared to males.")
print("- Higher-class passengers (Pclass=1) survived more than lower classes.")
print("- Younger passengers (children) had slightly higher chances of survival.")
print("- Strong correlation between Sex, Pclass, and Survival.")
