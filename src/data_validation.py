import pandas as pd
import os

def validate_dataset(filepath, output_dir):
    filename = os.path.basename(filepath)
    print(f"Loading dataset from: {filepath}")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Basic metrics
    num_rows, num_cols = df.shape
    columns = df.columns.tolist()
    data_types = df.dtypes.astype(str).to_dict()
    missing_values = df.isnull().sum().to_dict()
    duplicates = df.duplicated().sum()
    unique_values = df.nunique().to_dict()
    
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    stat_summary = df.describe(include='all').to_string()
    
    # Output path
    report_path = os.path.join(output_dir, "validation_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Dataset Filename: {filename}\n")
        f.write(f"Number of Rows: {num_rows}\n")
        f.write(f"Number of Columns: {num_cols}\n\n")
        
        f.write("--- Columns & Data Types ---\n")
        for col, dtype in data_types.items():
            f.write(f"{col}: {dtype}\n")
            
        f.write("\n--- Missing Values ---\n")
        for col, missing in missing_values.items():
            f.write(f"{col}: {missing}\n")
            
        f.write(f"\nDuplicate Records: {duplicates}\n\n")
        
        f.write("--- Unique Values (Cardinality) ---\n")
        for col, unique in unique_values.items():
            f.write(f"{col}: {unique}\n")
            
        f.write("\n--- Feature Types ---\n")
        f.write(f"Numerical Columns: {', '.join(numerical_cols)}\n")
        f.write(f"Categorical Columns: {', '.join(categorical_cols)}\n\n")
        
        f.write("--- Statistical Summary ---\n")
        f.write(stat_summary)
        f.write("\n")
        
    print(f"Validation report saved to {report_path}")
    print(f"Rows: {num_rows}, Columns: {num_cols}")
    print(f"Columns: {columns}")
    
if __name__ == "__main__":
    filepath = os.path.join("data", "faculty_dataset1.csv")
    output_dir = os.path.join("outputs", "eda")
    os.makedirs(output_dir, exist_ok=True)
    validate_dataset(filepath, output_dir)
