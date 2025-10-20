import pandas as pd
import numpy as np
import kagglehub
import os

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
print("Dataset downloaded to:", path)
file_path = os.path.join(path, "marketing_campaign.csv")
df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
print("\nDataset loaded successfully!\n")

print("Dataset Overview")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\n")

#Handle missing values
#Fill missing values 
df['Income'] = df['Income'].fillna(df['Income'].median()) 
print(" Missing values handled.\n")

#Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f" Removed {before - after} duplicate rows.\n")

# Standardize column names
df.columns = (
    df.columns
    .str.strip()             
    .str.lower()              
    .str.replace(' ', '_')    
    .str.replace('-', '_')    )
print(" Column names standardized.\n")

# Standardize categorical text
if 'education' in df.columns:
    df['education'] = df['education'].str.title().str.strip()

if 'marital_status' in df.columns:
    df['marital_status'] = df['marital_status'].str.title().str.strip()

print("Text columns standardized.\n")

#  Convert date formats 
if 'dt_customer' in df.columns:
    df['dt_customer'] = pd.to_datetime(df['dt_customer'], errors='coerce')
    print("Date column converted to datetime.\n")

#  Fix data types
if 'year_birth' in df.columns:
    df['age'] = 2025 - df['year_birth']  
    df['age'] = df['age'].astype(int)
print("Data types verified and updated.\n")

#  Outlier treatment 
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper, upper,
                       np.where(df[col] < lower, lower, df[col]))
print("Outliers capped using IQR method.\n")

# Export cleaned dataset
output_path = "cleaned_customer_personality.csv"
df.to_csv(output_path, index=False)
print(f" Cleaned dataset saved as '{output_path}'\n")

# Summary of changes
print("DATA CLEANING SUMMARY")
print(f"Final shape: {df.shape}")
print(f"Missing values (after cleaning):\n{df.isnull().sum().sum()} total missing values")
print("Duplicates removed:", before - after)
print("Columns after cleaning:\n", df.columns.tolist())
print("\n")
print(" Data cleaning and preprocessing completed successfully!")
