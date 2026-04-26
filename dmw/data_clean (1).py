#20240802151 and #20240802131

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
print("Initial shape:", df.shape)

# Check basic info
print(df.info())
print(df.describe())

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Fill missing values (safety step)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate rows
df = df.drop_duplicates()
print("After duplicates:", df.shape)

# Remove outliers using IQR (only key columns)
outlier_cols = ["Age", "Height", "Weight"]

Q1 = df[outlier_cols].quantile(0.25)
Q3 = df[outlier_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[outlier_cols] < (Q1 - 1.5 * IQR)) |
          (df[outlier_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("After outliers:", df.shape)

# Create BMI feature (important health indicator)
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Encode categorical columns to numeric
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Simple visualization (Age vs Height)
df.plot(x='Age', y='Height', style='o')
plt.title("Age vs Height")
plt.show()

# Final check
print(df.info())
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned.csv", index=False)
print("✅ Cleaned data saved")