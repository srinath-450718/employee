import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def build_model(input_path, model_dir, output_dir):
    print(f"Loading data from {input_path} for Modeling...")
    df = pd.read_csv(input_path)
    
    # 1. Feature Selection
    # The target is 'Target' (High, Medium, Low) - Classification
    # To avoid target leakage, we exclude performance proxies like 
    # 'Student_Feedback', 'Course_Completion_Rate', 'Pass_Percentage', 'Collaboration_Score', 'Innovation_Score'.
    # We focus on demographics, experience, and training variables.
    
    features = [
        'Age', 'Teaching_Experience', 'Industry_Experience', 
        'Publications', 'Certification_Count', 'Workshops_Attended', 
        'Training_Hours', 'FDP_Attended', 'Online_Courses', 
        'Skill_Upgrade_Score', 'Department', 'Highest_Degree'
    ]
    
    target = 'Target'
    
    X = df[features].copy()
    y = df[target]
    
    # 3. Preprocessing
    # Encode categorical features
    label_encoders = {}
    for col in ['Department', 'Highest_Degree']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Optional scaling (not strictly required for Decision Trees, but good practice if extending later)
    # We'll stick to raw values for interpretability of the Decision Tree
    
    # 4. Model Training
    print("Training DecisionTreeClassifier...")
    dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
    dt_classifier.fit(X_train, y_train)
    
    # 5. Prediction & Validation
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # 6. Feature Importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt_classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Save Model
    model_path = os.path.join(model_dir, "dt_model.pkl")
    joblib.dump(dt_classifier, model_path)
    
    # Save encoders for later use in app
    encoder_path = os.path.join(model_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoder_path)
    
    # Save Report
    report_path = os.path.join(output_dir, "model_report.txt")
    with open(report_path, "w") as f:
        f.write("--- DECISION TREE CLASSIFICATION REPORT ---\n\n")
        f.write(f"Algorithm: DecisionTreeClassifier\n")
        f.write(f"Target Variable: {target} (Classification)\n\n")
        
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n\n")
        
        f.write("Feature Importances:\n")
        f.write(feature_importances.to_string(index=False))
        f.write("\n")
        
    print(f"Model saved to: {model_path}")
    print(f"Model Report saved to: {report_path}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    input_path = os.path.join("outputs", "cleaned_data", "featured_dataset.csv")
    model_dir = "models"
    output_dir = os.path.join("outputs", "model_results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    build_model(input_path, model_dir, output_dir)
