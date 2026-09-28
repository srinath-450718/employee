import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations(input_path, output_dir):
    print(f"Loading data from {input_path} for Visualization...")
    df = pd.read_csv(input_path)
    
    # Setup visualization style
    sns.set_theme(style="whitegrid")
    
    # Helper to save figures
    def save_fig(fig, filename):
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved: {filepath}")

    # 1. Training Hours vs Performance Target
    # We'll use a boxplot to show the distribution of Training Hours across Target categories
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='Target', y='Training_Hours', order=['Low', 'Medium', 'High'], ax=ax1, palette='Set2')
    ax1.set_title("Distribution of Training Hours by Performance Target", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Performance Target", fontsize=12)
    ax1.set_ylabel("Training Hours", fontsize=12)
    save_fig(fig1, "1_training_hours_vs_target.png")
    
    # 2. Workshops, FDP, Online Courses vs Performance (Target)
    # Melting the dataframe to plot multiple activities side-by-side
    activity_cols = ['Workshops_Attended', 'FDP_Attended', 'Online_Courses']
    df_melt = pd.melt(df, id_vars=['Target'], value_vars=activity_cols, 
                      var_name='Activity_Type', value_name='Count')
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Activity_Type', y='Count', hue='Target', 
                hue_order=['Low', 'Medium', 'High'], ax=ax2, palette='viridis', errorbar=None)
    ax2.set_title("Average Training Activities by Performance Target", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Training Activity Type", fontsize=12)
    ax2.set_ylabel("Average Count", fontsize=12)
    ax2.legend(title="Target", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Clean up x-labels for better readability
    x_labels = [label.get_text().replace('_', ' ') for label in ax2.get_xticklabels()]
    ax2.set_xticklabels(x_labels)
    save_fig(fig2, "2_activities_vs_target.png")
    
    # 3. Training Intensity by Department
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Department', y='Professional_Development_Intensity', 
                ax=ax3, palette='mako', errorbar=None)
    ax3.set_title("Average Professional Development Intensity by Department", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Department", fontsize=12)
    ax3.set_ylabel("Professional Dev Intensity", fontsize=12)
    save_fig(fig3, "3_intensity_by_department.png")
    
    # 4. Correlation Heatmap (Continuous variables)
    cols_for_heatmap = ['Training_Hours', 'Workshops_Attended', 'FDP_Attended', 
                        'Online_Courses', 'Skill_Upgrade_Score', 'Student_Feedback', 
                        'Course_Completion_Rate', 'Pass_Percentage', 'Total_Training_Exposure']
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    corr_matrix = df[cols_for_heatmap].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.1, vmax=0.1, ax=ax4, 
                cbar_kws={'label': 'Correlation Coefficient'})
    ax4.set_title("Correlation Heatmap: Training vs Performance", fontsize=14, fontweight='bold')
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    save_fig(fig4, "4_correlation_heatmap.png")
    
    # 5. Skill Upgrade Score Distribution by Department
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=df, x='Department', y='Skill_Upgrade_Score', palette='pastel', ax=ax5)
    ax5.set_title("Distribution of Skill Upgrade Scores by Department", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Department", fontsize=12)
    ax5.set_ylabel("Skill Upgrade Score", fontsize=12)
    save_fig(fig5, "5_skill_upgrade_by_dept.png")

if __name__ == "__main__":
    input_path = os.path.join("outputs", "cleaned_data", "featured_dataset.csv")
    output_dir = os.path.join("outputs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    create_visualizations(input_path, output_dir)
