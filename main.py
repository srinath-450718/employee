import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Executing {script_path}...")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR executing {script_path}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        print(f"SUCCESS: {script_path} completed.\n")

def main():
    print("Starting Training & Faculty Performance Effectiveness Analysis Pipeline...\n")
    
    scripts_to_run = [
        "src/data_validation.py",
        "src/data_cleaning.py",
        "src/eda.py",
        "src/feature_engineering.py",
        "src/training_analysis.py",
        "src/confounding_analysis.py",
        "src/visualization.py",
        "src/model.py",
        "src/evaluation.py"
    ]
    
    for script in scripts_to_run:
        run_script(script)
        
    print("==================================================")
    print("ALL MODULES EXECUTED SUCCESSFULLY!")
    print("Project pipeline is complete and verified.")
    print("You can now launch the Streamlit dashboard using: streamlit run app.py")
    print("==================================================")

if __name__ == "__main__":
    main()
