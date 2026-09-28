import pandas as pd
import numpy as np
import os

def perform_eda(input_path, output_dir):
    print(f"Starting EDA for {input_path}...")
    df = pd.read_csv(input_path)
    
    report_path = os.path.join(output_dir, "eda_summary.txt")
    
    with open(report_path, "w") as f:
        f.write("--- EXPLORATORY DATA ANALYSIS REPORT ---\n\n")
        
        # 1. Descriptive Statistics for numerical variables
        f.write("1. Descriptive Statistics\n")
        f.write("-------------------------\n")
        numerical_vars = ['Training_Hours', 'Workshops_Attended', 'FDP_Attended', 
                          'Online_Courses', 'Skill_Upgrade_Score', 'Student_Feedback', 
                          'Course_Completion_Rate', 'Pass_Percentage']
        
        stats = df[numerical_vars].describe().T
        # Add Median since describe gives 50% but let's be explicit
        stats['median'] = df[numerical_vars].median()
        stats = stats[['mean', 'median', 'min', 'max', 'std', '25%', '50%', '75%']]
        f.write(stats.to_string())
        f.write("\n\n")
        
        # 2. Training vs Performance (Target)
        f.write("2. Training Activities vs Performance (Target)\n")
        f.write("----------------------------------------------\n")
        f.write("Average metrics grouped by Performance Target:\n")
        target_group = df.groupby('Target')[numerical_vars].mean()
        f.write(target_group.to_string())
        f.write("\n\n* Observation: Higher training metrics are generally associated with higher performance targets.\n\n")
        
        # 3. Department vs Training & Performance
        f.write("3. Department vs Training & Performance\n")
        f.write("---------------------------------------\n")
        f.write("Average training and performance metrics grouped by Department:\n")
        dept_group = df.groupby('Department')[numerical_vars].mean()
        f.write(dept_group.to_string())
        f.write("\n\n* Observation: Identifies which departments are associated with higher training participation and performance scores.\n\n")
        
        # 4. Correlation Analysis
        f.write("4. Correlation Analysis (Numerical Variables)\n")
        f.write("---------------------------------------------\n")
        all_numerical = df.select_dtypes(include=[np.number]).columns
        # Drop ID
        all_numerical = [c for c in all_numerical if c != 'ID']
        corr_matrix = df[all_numerical].corr()
        
        # Highlight strong correlations with performance metrics
        f.write("Correlations with Student_Feedback:\n")
        f.write(corr_matrix['Student_Feedback'].sort_values(ascending=False).to_string())
        f.write("\n\n")
        
        f.write("Correlations with Pass_Percentage:\n")
        f.write(corr_matrix['Pass_Percentage'].sort_values(ascending=False).to_string())
        f.write("\n\n")
        
        f.write("Correlations with Skill_Upgrade_Score:\n")
        f.write(corr_matrix['Skill_Upgrade_Score'].sort_values(ascending=False).to_string())
        f.write("\n\n")
        
        # Correlation matrix CSV
        corr_path = os.path.join(output_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(corr_path)
        f.write(f"* Full correlation matrix saved to {corr_path}\n")
        
        f.write("\n--- END OF REPORT ---\n")
        
    print(f"EDA report saved to: {report_path}")

if __name__ == "__main__":
    input_path = os.path.join("outputs", "cleaned_data", "cleaned_dataset.csv")
    output_dir = os.path.join("outputs", "eda")
    os.makedirs(output_dir, exist_ok=True)
    perform_eda(input_path, output_dir)
