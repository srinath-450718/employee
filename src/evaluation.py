import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(data_path, model_dir, output_dir):
    print(f"Loading data from {data_path} and model from {model_dir} for Evaluation...")
    df = pd.read_csv(data_path)
    
    # Load model and encoders
    model_path = os.path.join(model_dir, "dt_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoders.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found. Please run Module 7 first.")
        return
        
    dt_classifier = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)
    
    # Same feature set and split as Module 7
    features = [
        'Age', 'Teaching_Experience', 'Industry_Experience', 
        'Publications', 'Certification_Count', 'Workshops_Attended', 
        'Training_Hours', 'FDP_Attended', 'Online_Courses', 
        'Skill_Upgrade_Score', 'Department', 'Highest_Degree'
    ]
    target = 'Target'
    
    X = df[features].copy()
    y = df[target]
    
    for col in ['Department', 'Highest_Degree']:
        X[col] = label_encoders[col].transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Predictions
    y_pred = dt_classifier.predict(X_test)
    
    # 2. Calculate Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    # Using weighted average for multiclass precision/recall/f1
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # 3. Feature Importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt_classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Save Report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("--- DECISION TREE EVALUATION REPORT ---\n\n")
        f.write("Task: Classification\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f} (Weighted)\n")
        f.write(f"Recall:    {recall:.4f} (Weighted)\n")
        f.write(f"F1-Score:  {f1:.4f} (Weighted)\n\n")
        
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix))
        f.write("\n\n")
        
        f.write("--- FEATURE IMPORTANCE ---\n")
        f.write("IMPORTANT NOTE: Feature importance indicates predictive contribution to this specific model. ")
        f.write("It DOES NOT prove causation. A highly important feature helps the model split the data efficiently, ")
        f.write("but does not necessarily mean altering that feature will cause a direct change in the target outcome.\n\n")
        
        f.write(feature_importances.to_string(index=False))
        f.write("\n")
        
    print(f"Evaluation report saved to: {report_path}")
    
    # 4. Visualizations
    sns.set_theme(style="whitegrid")
    
    # Plot Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=dt_classifier.classes_, yticklabels=dt_classifier.classes_)
    ax1.set_title("Confusion Matrix", fontweight='bold')
    ax1.set_ylabel("Actual Label")
    ax1.set_xlabel("Predicted Label")
    conf_path = os.path.join(output_dir, "confusion_matrix.png")
    fig1.savefig(conf_path, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    
    # Plot Feature Importance
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis', ax=ax2)
    ax2.set_title("Decision Tree Feature Importance", fontweight='bold')
    ax2.set_xlabel("Relative Importance")
    ax2.set_ylabel("Feature")
    # Add textual note inside the plot to prevent misinterpretation
    plt.figtext(0.5, -0.05, "Note: Importance indicates predictive contribution, not causation.", 
                ha="center", fontsize=10, style='italic', color='red')
    feat_path = os.path.join(output_dir, "feature_importance.png")
    fig2.savefig(feat_path, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    
    print(f"Saved: {conf_path}")
    print(f"Saved: {feat_path}")
    print("Evaluation Complete.")

if __name__ == "__main__":
    data_path = os.path.join("outputs", "cleaned_data", "featured_dataset.csv")
    model_dir = "models"
    output_dir = os.path.join("outputs", "model_results")
    os.makedirs(output_dir, exist_ok=True)
    evaluate_model(data_path, model_dir, output_dir)
