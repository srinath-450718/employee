import pandas as pd
import numpy as np
import os

def clean_data(input_path, output_dir):
    print(f"Starting data cleaning for {input_path}...")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    original_rows, original_cols = df.shape
    
    # 2. Missing Value Analysis
    missing_before = df.isnull().sum().sum()
    
    # Handling missing values (though we expect 0, this makes it robust)
    # We will drop rows with missing values if any exist for simplicity, 
    # or fill with median for numeric, mode for categorical.
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
    missing_after = df.isnull().sum().sum()
    
    # 3. Duplicate Detection & Removal
    duplicates_found = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    
    # 4. Data Type Correction & Invalid Values Detection
    invalid_values = {}
    
    # Validating categorical values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        invalid_values[col] = f"Unique values: {df[col].unique().tolist()}"
        
    # Numerical Range Validation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val < 0 and col not in ['ID']: # Assuming no negative values in our context
             invalid_values[col] = f"Contains negative values. Min: {min_val}"
             
    # 5. Outlier Inspection (using IQR method)
    outliers_report = {}
    for col in numerical_cols:
        if col != 'ID': # Skip ID
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            if outlier_count > 0:
                outliers_report[col] = outlier_count
                
    # 6. Save Cleaned Dataset
    final_rows, final_cols = df.shape
    records_removed = original_rows - final_rows
    
    cleaned_data_path = os.path.join(output_dir, "cleaned_dataset.csv")
    df.to_csv(cleaned_data_path, index=False)
    
    # 7. Generate Cleaning Report
    report_path = os.path.join(output_dir, "cleaning_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Data Cleaning Report ---\n\n")
        f.write(f"Original Rows: {original_rows}\n")
        f.write(f"Final Rows: {final_rows}\n")
        f.write(f"Records Removed: {records_removed}\n\n")
        
        f.write(f"Missing Values Before Cleaning: {missing_before}\n")
        f.write(f"Missing Values After Cleaning: {missing_after}\n\n")
        
        f.write(f"Duplicate Records Found: {duplicates_found}\n\n")
        
        f.write("--- Final Data Types ---\n")
        for col, dtype in df.dtypes.items():
            f.write(f"{col}: {dtype}\n")
            
        f.write("\n--- Categorical & Range Validations ---\n")
        for col, msg in invalid_values.items():
            f.write(f"{col}: {msg}\n")
            
        f.write("\n--- Outlier Inspection (Not Removed) ---\n")
        if outliers_report:
            for col, count in outliers_report.items():
                f.write(f"{col}: {count} potential outliers detected\n")
        else:
            f.write("No potential outliers detected using IQR method.\n")
            
    print(f"Cleaned dataset saved to: {cleaned_data_path}")
    print(f"Cleaning report saved to: {report_path}")

if __name__ == "__main__":
    input_path = os.path.join("data", "faculty_dataset1.csv")
    output_dir = os.path.join("outputs", "cleaned_data")
    os.makedirs(output_dir, exist_ok=True)
    clean_data(input_path, output_dir)
