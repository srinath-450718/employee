import pandas as pd
import numpy as np
import os

def engineer_features(input_path, output_dir):
    print(f"Loading data from {input_path} for Feature Engineering...")
    df = pd.read_csv(input_path)
    
    # ---------------------------------------------------------
    # Feature 1: Training_Activity_Count
    # Variables used: Workshops_Attended, FDP_Attended, Online_Courses, Certification_Count
    # Formula: Workshops_Attended + FDP_Attended + Online_Courses + Certification_Count
    # Reasoning: Provides a single cumulative metric of all discrete professional development events attended.
    # ---------------------------------------------------------
    df['Training_Activity_Count'] = (
        df['Workshops_Attended'] + 
        df['FDP_Attended'] + 
        df['Online_Courses'] + 
        df['Certification_Count']
    )
    
    # ---------------------------------------------------------
    # Feature 2: Total_Training_Exposure
    # Variables used: Training_Hours, Training_Activity_Count
    # Formula: Training_Hours + (Training_Activity_Count * 5)
    # Reasoning: Combines continuous training hours with discrete activities. Assuming each activity roughly 
    # contributes an equivalent of 5 'exposure units' or hours to their baseline training hours.
    # ---------------------------------------------------------
    df['Total_Training_Exposure'] = df['Training_Hours'] + (df['Training_Activity_Count'] * 5)
    
    # ---------------------------------------------------------
    # Feature 3: Professional_Development_Intensity
    # Variables used: Total_Training_Exposure, Teaching_Experience
    # Formula: Total_Training_Exposure / (Teaching_Experience + 1)
    # Reasoning: Normalizes training exposure against years of teaching experience. 
    # Shows how intensely a faculty member trains relative to their career length. (+1 prevents division by zero).
    # ---------------------------------------------------------
    df['Professional_Development_Intensity'] = df['Total_Training_Exposure'] / (df['Teaching_Experience'] + 1)
    
    # ---------------------------------------------------------
    # Feature 4: Training_Participation_Indicator
    # Variables used: Training_Activity_Count
    # Formula: 1 if Training_Activity_Count > median(Training_Activity_Count) else 0
    # Reasoning: A binary flag that easily identifies faculty who participate in more training activities 
    # than the average peer (useful for classification splits).
    # ---------------------------------------------------------
    median_activities = df['Training_Activity_Count'].median()
    df['Training_Participation_Indicator'] = (df['Training_Activity_Count'] > median_activities).astype(int)
    
    # ---------------------------------------------------------
    # Feature 5: Training_Engagement_Category
    # Variables used: Total_Training_Exposure
    # Formula: pd.qcut into 3 quantiles ('Low', 'Medium', 'High')
    # Reasoning: Transforms the continuous exposure metric into a categorical variable, 
    # making it easier to analyze group differences and build segment-based insights.
    # ---------------------------------------------------------
    df['Training_Engagement_Category'] = pd.qcut(
        df['Total_Training_Exposure'], 
        q=3, 
        labels=['Low', 'Medium', 'High']
    )
    
    # ---------------------------------------------------------
    # Feature 6: Skill_Application_Ratio
    # Variables used: Skill_Upgrade_Score, Total_Training_Exposure
    # Formula: Skill_Upgrade_Score / (Total_Training_Exposure + 1)
    # Reasoning: Measures how efficiently training exposure translates into an upgraded skill score. 
    # Identifies 'fast learners' or highly effective training attendees.
    # ---------------------------------------------------------
    df['Skill_Application_Ratio'] = df['Skill_Upgrade_Score'] / (df['Total_Training_Exposure'] + 1)

    
    # Save the engineered dataset
    output_file = os.path.join(output_dir, "featured_dataset.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Feature Engineering complete. Added 6 new features.")
    print(f"Engineered dataset saved to {output_file}")
    print("\nNew Features Sample:")
    print(df[['Training_Activity_Count', 'Total_Training_Exposure', 
              'Professional_Development_Intensity', 'Training_Participation_Indicator', 
              'Training_Engagement_Category', 'Skill_Application_Ratio']].head())

if __name__ == "__main__":
    input_path = os.path.join("outputs", "cleaned_data", "cleaned_dataset.csv")
    output_dir = os.path.join("outputs", "cleaned_data")
    os.makedirs(output_dir, exist_ok=True)
    engineer_features(input_path, output_dir)
