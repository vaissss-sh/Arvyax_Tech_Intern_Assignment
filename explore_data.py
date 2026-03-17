import pandas as pd
import json

train_path = "Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv"
test_path = "arvyax_test_inputs_120.xlsx - Sheet1.csv"

def analyze_df(df, name):
    print(f"--- Analysis for {name} ---")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nSample Data (first 3 rows):")
    for i, row in df.head(3).iterrows():
        print(f"Row {i}: {row.to_dict()}")
        
    for col in df.columns:
        if df[col].nunique() < 20:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts(dropna=False))
    print("\n" + "="*50 + "\n")

try:
    train_df = pd.read_csv(train_path)
    analyze_df(train_df, "Train Dataset")
except Exception as e:
    print(f"Error reading train dataset: {e}")

try:
    test_df = pd.read_csv(test_path)
    analyze_df(test_df, "Test Dataset")
except Exception as e:
    print(f"Error reading test dataset: {e}")
