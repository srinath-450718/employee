import pandas as pd
import os

def run_confounding_analysis(input_path, output_dir):
    print(f"Loading data from {input_path} for Confounding Factor Analysis...")
    df = pd.read_csv(input_path)
    
    report_path = os.path.join(output_dir, "confounding_analysis.txt")
    
    with open(report_path, "w") as f:
        f.write("========================================================\n")
        f.write("       CONFOUNDING FACTOR ANALYSIS REPORT\n")
        f.write("========================================================\n\n")
        
        # Mandatory Limitations Section
        f.write("--- IMPORTANT METHODOLOGICAL LIMITATIONS ---\n")
        f.write("1. Observational Data Limitations: This dataset is purely observational. Because it does not contain a randomized training assignment (A/B testing format) nor longitudinal pre/post-training measurements, CAUSAL training effectiveness cannot be established.\n")
        f.write("2. No Causal Claim: We DO NOT claim that training alone *causes* better performance. Any correlation identified is strictly an *observed association*.\n")
        f.write("3. Clarification of Terms:\n")
        f.write("   - Observed Association: Two variables move together (e.g., more training correlates with higher target scores).\n")
        f.write("   - Possible Confounding Factor: A third variable that influences BOTH the training activity and the performance outcome, making it look like training causes the performance when the confounder is actually the root driver.\n")
        f.write("   - Causal Conclusion: Stating 'A causes B'. This CANNOT be done with this dataset.\n\n")
        
        f.write("--- ANALYSIS OF POTENTIAL CONFOUNDING FACTORS ---\n\n")
        
        # 1. Department
        f.write("1. Department (Institutional Context)\n")
        f.write("- Why it's a confounder: Different departments have different budgets, expectations, and grading cultures. A department like 'Science' might naturally mandate more training (increasing training hours) while simultaneously having a lenient grading curve (increasing performance). The relationship between training and performance here might just be a proxy for departmental culture.\n")
        dept_train = df.groupby('Department')['Total_Training_Exposure'].mean()
        dept_perf = df.groupby('Department')['Course_Completion_Rate'].mean()
        f.write(f"- Observed:\n  Avg Exposure: {dept_train.to_dict()}\n  Avg Completion Rate: {dept_perf.to_dict()}\n\n")
        
        # 2. Experience
        f.write("2. Experience (Teaching & Industry)\n")
        f.write("- Why it's a confounder: Highly experienced faculty might be both more likely to attend specialized workshops (high training) AND be naturally better teachers due to tenure (high performance). Experience drives both, making training look like the cause of performance.\n")
        corr_exp = df[['Teaching_Experience', 'Total_Training_Exposure', 'Pass_Percentage']].corr()
        f.write(f"- Correlation Matrix:\n{corr_exp}\n\n")
        
        # 3. Workload Proxies (Team Projects, Mentorships)
        f.write("3. Workload (Team Projects & Mentorships)\n")
        f.write("- Why it's a confounder: Faculty with heavy workloads (high mentorship, high team projects) might not have time for training (low training hours) but might also suffer from burnout (low performance scores). Lack of time is the confounder suppressing both metrics.\n")
        corr_workload = df[['Team_Project_Count', 'Mentorship_Count', 'Training_Hours', 'Pass_Percentage']].corr()
        f.write(f"- Correlation Matrix:\n{corr_workload}\n\n")
        
        # 4. Skill Level (Skill Upgrade Score)
        f.write("4. Skill Level (Skill Upgrade Score)\n")
        f.write("- Why it's a confounder: Inherently highly motivated or skilled individuals are more likely to seek out training (high exposure) and are also naturally more likely to perform well on the job (high target). The underlying trait of 'Motivation/Skill' causes both.\n")
        corr_skill = df[['Skill_Upgrade_Score', 'Total_Training_Exposure', 'Student_Feedback']].corr()
        f.write(f"- Correlation Matrix:\n{corr_skill}\n\n")
        
        # 5. Course Completion & Student Feedback
        f.write("5. Student Feedback & Course Completion\n")
        f.write("- Why it's a confounder: These variables often act as proxies for the 'Target' performance. If a faculty member teaches a notoriously easy course, their course completion rate will be high regardless of their training. If they use their free time to train instead of prep, they might have high training but poor feedback.\n\n")
        
        f.write("--- CONCLUSION ---\n")
        f.write("The observed associations between training features and the performance Target must be viewed through the lens of these confounding factors. Training is a piece of the puzzle, but tenure, department culture, and baseline motivation are heavily intertwined confounders.\n")

    print(f"Confounding analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    input_path = os.path.join("outputs", "cleaned_data", "featured_dataset.csv")
    output_dir = os.path.join("outputs", "training_analysis")
    os.makedirs(output_dir, exist_ok=True)
    run_confounding_analysis(input_path, output_dir)
