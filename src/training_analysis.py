import pandas as pd
import numpy as np
import os

def analyze_training_effectiveness(input_path, output_dir):
    print(f"Loading data from {input_path} for Training Effectiveness Analysis...")
    df = pd.read_csv(input_path)
    
    report_path = os.path.join(output_dir, "training_effectiveness_report.txt")
    
    with open(report_path, "w") as f:
        f.write("--- TRAINING EFFECTIVENESS ANALYSIS REPORT ---\n\n")
        
        # Mandatory Limitation Statements
        f.write("IMPORTANT LIMITATIONS:\n")
        f.write("- A direct causal comparison between trained and untrained employees cannot be performed because the dataset does not contain a true 'trained-vs-untrained' control group indicator.\n")
        f.write("- Pre-training vs post-training improvement cannot be directly measured because the dataset does not contain longitudinal pre/post performance measurements.\n\n")
        
        f.write("--- ANALYSIS ---\n\n")
        
        # Helper list for performance columns
        perf_cols = ['Student_Feedback', 'Course_Completion_Rate', 'Pass_Percentage']
        
        # 1. Training hours vs performance
        f.write("1. Training Hours vs Performance\n")
        f.write(df[['Training_Hours'] + perf_cols].corr().loc[['Training_Hours'], perf_cols].to_string())
        f.write("\n\n")
        
        # 2. Workshops attended vs performance
        f.write("2. Workshops Attended vs Performance\n")
        f.write(df[['Workshops_Attended'] + perf_cols].corr().loc[['Workshops_Attended'], perf_cols].to_string())
        f.write("\n\n")
        
        # 3. FDP attendance vs performance
        f.write("3. FDP Attendance vs Performance\n")
        f.write(df[['FDP_Attended'] + perf_cols].corr().loc[['FDP_Attended'], perf_cols].to_string())
        f.write("\n\n")
        
        # 4. Online courses vs performance
        f.write("4. Online Courses vs Performance\n")
        f.write(df[['Online_Courses'] + perf_cols].corr().loc[['Online_Courses'], perf_cols].to_string())
        f.write("\n\n")
        
        # 5. Skill upgrade score vs performance
        f.write("5. Skill Upgrade Score vs Performance\n")
        f.write(df[['Skill_Upgrade_Score'] + perf_cols].corr().loc[['Skill_Upgrade_Score'], perf_cols].to_string())
        f.write("\n\n")
        
        # 6. Training activity intensity vs performance
        f.write("6. Professional Development Intensity vs Performance\n")
        if 'Professional_Development_Intensity' in df.columns:
            f.write(df[['Professional_Development_Intensity'] + perf_cols].corr().loc[['Professional_Development_Intensity'], perf_cols].to_string())
        else:
            f.write("Feature not found.")
        f.write("\n\n")
        
        # 7. Department-level training participation
        f.write("7. Department-Level Training Participation\n")
        train_cols = ['Training_Hours', 'Workshops_Attended', 'FDP_Attended', 'Online_Courses', 'Total_Training_Exposure']
        f.write(df.groupby('Department')[train_cols].mean().to_string())
        f.write("\n\n")
        
        # 8. Department-level performance
        f.write("8. Department-Level Performance\n")
        f.write(df.groupby('Department')[perf_cols].mean().to_string())
        f.write("\n\n")
        
        # 9. Training participation across performance categories
        f.write("9. Training Participation across Performance Categories (Target)\n")
        f.write(df.groupby('Target')[train_cols].mean().to_string())
        f.write("\n\n")
        
        f.write("--- END OF REPORT ---\n")
        
    print(f"Training Effectiveness Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    input_path = os.path.join("outputs", "cleaned_data", "featured_dataset.csv")
    output_dir = os.path.join("outputs", "training_analysis")
    os.makedirs(output_dir, exist_ok=True)
    analyze_training_effectiveness(input_path, output_dir)
