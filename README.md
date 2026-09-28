# HR — Does Training Actually Improve Performance?

## 1. Project Title
**Training Effectiveness Analytics**

## 2. Problem Statement
Many corporate and academic institutions spend heavily on employee training and Faculty Development Programmes (FDPs). This project aims to determine mathematically if this training actually translates to better performance in the classroom or workplace.

## 3. Objective
To evaluate whether employee training programmes (workshops, online courses, FDPs) are associated with improved performance targets, and to identify the hidden confounders affecting these outcomes.

## 4. Industry
Corporate HR / Academic Administration

## 5. Dataset Description
The dataset contains records of faculty members, detailing their demographic information, training exposure, academic performance, and a final categorized Performance `Target` (High, Medium, Low). 

## 6. Features Used
- `Training_Hours`, `Workshops_Attended`, `FDP_Attended`, `Online_Courses`
- `Student_Feedback`, `Pass_Percentage`, `Course_Completion_Rate`
- `Skill_Upgrade_Score`
- `Department`, `Teaching_Experience`, `Publications`

## 7. Data Preprocessing
- Removed duplicate records.
- Imputed missing numerical values using medians to avoid outlier skewing.
- Imputed categorical missing values using the statistical mode.

## 8. Exploratory Data Analysis
Comprehensive descriptive statistical analysis was run to determine the distributions, variances, and baseline averages of training behaviors across the population.

## 9. Training Effectiveness Analysis
We analyzed the correlation between training exposure and final output. Notably, linear volume (pure training hours) lacked a direct linear association with elite performance metrics.

## 10. Course/Program Analysis
Compared the effectiveness of Workshops vs. FDPs vs. Online Courses. Applied, active learning yielded stronger associations with performance than passive hours.

## 11. Employee Segment Analysis
Grouped and analyzed employees by Department and Tenure.

## 12. Confounding Factors
Identified massive third-variable influencers:
1. **Department Culture:** Variances in grading leniency and mandatory training.
2. **Experience Level:** Naturally drives both higher performance and FDP attendance.
3. **Inherent Motivation:** A hidden trait driving both the `Skill_Upgrade_Score` and the performance outcome.

## 13. Visualizations
- Training Hours vs Target Performance (Boxplot)
- Student Feedback vs Target Performance (Boxplot)
- Skill Upgrades by Department (Violin Plot)
- Correlation Heatmap

## 14. Decision Tree Model
A `DecisionTreeClassifier` was trained to classify the performance `Target` based strictly on non-leaking demographic and training variables.

## 15. Model Evaluation
- **Accuracy:** ~39.75%
- Weighted F1-Score: 38.83%
- Evaluated via Confusion Matrix and detailed Classification Reports.

## 16. Key Insights
1. **Quality over Volume:** Raw hour counts lack a linear association with elite performance.
2. **Applied Skill is Paramount:** `Skill_Upgrade_Score` was the absolute strongest predictor of performance (29.38% importance).
3. **Student Feedback Gap:** Top performers vastly outscored bottom performers in feedback regardless of training.
4. **Departmental Variation Confounds Metrics:** Different faculties have wildly different baselines.
5. **Experience is the Ultimate Substitute:** Real-world tenure outweighs formal FDPs.
6. **Performance is Complex:** A ~40% model accuracy proves human capability is influenced heavily by unmeasured, intrinsic traits.

## 17. Practical Action Plan
- Shift focus from generic hourly quotas to applied, project-based skill development.
- Standardize cross-departmental pedagogy.
- Rely on internal mentorship (pairing junior faculty with tenured high-performers).
- Implement strict pre/post measurement architectures for future training cohorts.

## 18. Technologies Used
- **Python:** Data processing backend.
- **Pandas, NumPy:** Data manipulation.
- **Matplotlib, Seaborn:** Data visualization.
- **Scikit-Learn:** Machine Learning (Decision Tree).
- **Streamlit:** Interactive UI/UX Dashboard.

## 19. Project Structure
```text
├── app.py               # Streamlit Dashboard UI
├── main.py              # Execution Orchestrator
├── requirements.txt     # Python Dependencies
├── src/                 # Backend analytical modules
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── training_analysis.py
│   ├── confounding_analysis.py
│   ├── visualization.py
│   ├── model.py
│   └── evaluation.py
└── models/              # Saved serialized models
```

## 20. Installation Instructions
1. Clone this repository.
2. Ensure Python 3.9+ is installed.
3. Run `pip install -r requirements.txt`.

## 21. How to Run
To run the full backend data pipeline and re-train the models:
```bash
python main.py
```

## 22. Streamlit Dashboard Instructions
To launch the interactive presentation dashboard:
```bash
streamlit run app.py
```
Upload your CSV through the sidebar to dynamically populate the UI.

## 23. ⚠️ Dataset Limitations
This dataset does not contain longitudinal pre-training and post-training performance measurements. Therefore, direct before-and-after training improvement **cannot be causally established**. The analysis evaluates *observed associations* between training exposure and observed performance rather than proving absolute causality.

## 24. Author
DS_Day01_49
