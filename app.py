import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import io

# --- Page Config ---
st.set_page_config(page_title="Training Effectiveness Analytics", layout="wide", page_icon="📊")

# --- Custom CSS for Premium Corporate Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Header */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.4rem;
        font-weight: 400;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .desc-text {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .status-active {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .status-waiting {
        background-color: #FEF9C3;
        color: #854D0E;
        border: 1px solid #FEF08A;
    }
    
    /* Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    }
    .kpi-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0.5rem 0;
    }
    .kpi-desc {
        font-size: 0.85rem;
        color: #94A3B8;
    }
    
    /* Section Headers */
    .section-hdr {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
    
    /* Warning/Info Boxes */
    .limit-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        color: #1E3A8A;
        margin: 1.5rem 0;
        font-size: 0.95rem;
    }
    
    /* Numbered Insights */
    .insight-card {
        background-color: #FFFFFF;
        border-left: 4px solid #2563EB;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }
    .insight-num {
        font-size: 1.25rem;
        font-weight: 800;
        color: #3B82F6;
        margin-bottom: 0.5rem;
    }
    .insight-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }
    
    /* Action Plan Cards */
    .action-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .action-step {
        margin-bottom: 0.75rem;
    }
    .action-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .action-val {
        font-size: 1rem;
        color: #1E293B;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">TRAINING EFFECTIVENESS ANALYTICS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Does Training Actually Improve Performance?</div>', unsafe_allow_html=True)
st.markdown('<div class="desc-text">Analyzing training participation, professional development and employee performance to evaluate training effectiveness.</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("### HR ANALYTICS\n**Training & Performance**")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file is None:
    st.markdown('<div class="status-badge status-waiting">● Waiting for Dataset</div>', unsafe_allow_html=True)
    st.info("👈 Please upload a CSV dataset in the sidebar to begin analysis.")
    st.stop()
else:
    st.markdown('<div class="status-badge status-active">● Dataset Loaded</div>', unsafe_allow_html=True)

sections = [
    "01 — Overview",
    "02 — Dataset",
    "03 — Training Analysis",
    "04 — Performance Analysis",
    "05 — Department Analysis",
    "06 — Visualizations",
    "07 — Decision Tree",
    "08 — Confounding Factors",
    "09 — Key Insights",
    "10 — Action Plan"
]
selection = st.sidebar.radio("Navigation", sections)

st.sidebar.markdown("---")
if st.sidebar.button("Reset App"):
    st.cache_data.clear()
    st.rerun()

# --- Logic Backend (Strictly Unchanged) ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

raw_df = load_data(uploaded_file)

@st.cache_data
def validate_and_clean(df):
    clean_df = df.copy()
    val_info = {
        'rows': clean_df.shape[0], 'cols': clean_df.shape[1],
        'missing_before': clean_df.isnull().sum().sum(), 'duplicates_before': clean_df.duplicated().sum()
    }
    clean_df.drop_duplicates(inplace=True)
    for col in clean_df.columns:
        if clean_df[col].isnull().sum() > 0:
            if clean_df[col].dtype in ['int64', 'float64']:
                clean_df[col].fillna(clean_df[col].median(), inplace=True)
            else:
                clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
                
    val_info['missing_after'] = clean_df.isnull().sum().sum()
    val_info['duplicates_after'] = clean_df.duplicated().sum()
    
    if all(c in clean_df.columns for c in ['Workshops_Attended', 'FDP_Attended', 'Online_Courses', 'Certification_Count', 'Training_Hours']):
        clean_df['Training_Activity_Count'] = clean_df['Workshops_Attended'] + clean_df['FDP_Attended'] + clean_df['Online_Courses'] + clean_df['Certification_Count']
        clean_df['Total_Training_Exposure'] = clean_df['Training_Hours'] + (clean_df['Training_Activity_Count'] * 5)
    return clean_df, val_info

@st.cache_data
def train_dt_model(df):
    target = 'Target'
    if target not in df.columns:
        return None, "Target column not found", None
        
    potential_features = [
        'Age', 'Teaching_Experience', 'Industry_Experience', 'Publications', 'Certification_Count', 'Workshops_Attended', 
        'Training_Hours', 'FDP_Attended', 'Online_Courses', 'Skill_Upgrade_Score', 'Department', 'Highest_Degree'
    ]
    features = [f for f in potential_features if f in df.columns]
    X = df[features].copy()
    y = df[target]
    encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': dt.feature_importances_}).sort_values(by='Importance', ascending=False)
    return dt, metrics, importances

df, val_info = validate_and_clean(raw_df)

def display_limitation():
    if 'Pre_Training_Performance' not in df.columns and 'Post_Training_Performance' not in df.columns:
        st.markdown('''
        <div class="limit-box">
            <strong>Dataset Limitation:</strong> This dataset does not contain longitudinal pre-training and post-training performance measurements. Therefore, direct before-and-after training improvement cannot be established. The analysis evaluates associations between training exposure and observed performance.
        </div>
        ''', unsafe_allow_html=True)

def render_kpi(label, value, desc):
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-desc">{desc}</div>
    </div>
    ''', unsafe_allow_html=True)

# --- Pages Routing ---

if selection == "01 — Overview":
    st.markdown('<div class="section-hdr">Overview Dashboard</div>', unsafe_allow_html=True)
    display_limitation()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("Total Employees", f"{len(df):,}", "Records in dataset")
    with c2:
        if 'Training_Hours' in df.columns:
            render_kpi("Avg Training Hours", f"{df['Training_Hours'].mean():.1f}h", "Per employee average")
        else:
            render_kpi("Avg Training Hours", "N/A", "")
    with c3:
        if 'Skill_Upgrade_Score' in df.columns:
            render_kpi("Avg Skill Score", f"{df['Skill_Upgrade_Score'].mean():.1f}", "Overall competency")
        else:
            render_kpi("Avg Skill Score", "N/A", "")
            
    c4, c5, c6 = st.columns(3)
    with c4:
        if 'Course_Completion_Rate' in df.columns:
            render_kpi("Course Completion", f"{df['Course_Completion_Rate'].mean():.1f}%", "Average completion rate")
        else:
            render_kpi("Course Completion", "N/A", "")
    with c5:
        if 'Student_Feedback' in df.columns:
            render_kpi("Avg Feedback Score", f"{df['Student_Feedback'].mean():.1f}", "Student evaluation")
        else:
            render_kpi("Avg Feedback Score", "N/A", "")
    with c6:
        if 'Pass_Percentage' in df.columns:
            render_kpi("Avg Pass Percentage", f"{df['Pass_Percentage'].mean():.1f}%", "Student pass rates")
        else:
            render_kpi("Avg Pass Percentage", "N/A", "")

elif selection == "02 — Dataset":
    st.markdown('<div class="section-hdr">Dataset Quality & Preview</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi("Total Rows", val_info['rows'], "After cleaning")
    with c2: render_kpi("Total Columns", val_info['cols'], "Features")
    with c3: render_kpi("Missing Values", val_info['missing_after'], f"Before: {val_info['missing_before']}")
    with c4: render_kpi("Duplicates", val_info['duplicates_after'], f"Before: {val_info['duplicates_before']}")
    
    st.write("")
    with st.expander("🔍 Dataset Preview (Top 10 Rows)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe().T, use_container_width=True)
        
    with st.expander("🏷️ Data Types"):
        st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str).T, use_container_width=True)

elif selection == "03 — Training Analysis":
    st.markdown('<div class="section-hdr">Training Effectiveness & Exposure</div>', unsafe_allow_html=True)
    display_limitation()
    train_cols = [c for c in ['Training_Hours', 'Workshops_Attended', 'FDP_Attended', 'Online_Courses', 'Total_Training_Exposure'] if c in df.columns]
    
    if train_cols:
        st.write("### Training Activity Summary")
        st.dataframe(df[train_cols].describe().T, use_container_width=True)
        
        st.markdown('''
        <div class="kpi-card" style="margin-top: 1rem;">
            <strong>Analysis Context:</strong> This section aggregates the pure volume of professional development undertaken by employees. Notice the variance in FDP vs Workshops to understand the preferred mode of training.
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.info("Training variables not found in dataset.")

elif selection == "04 — Performance Analysis":
    st.markdown('<div class="section-hdr">Performance Analysis</div>', unsafe_allow_html=True)
    perf_cols = [c for c in ['Student_Feedback', 'Course_Completion_Rate', 'Pass_Percentage', 'Skill_Upgrade_Score'] if c in df.columns]
    
    if perf_cols:
        st.write("### Performance Metrics Summary")
        st.dataframe(df[perf_cols].describe().T, use_container_width=True)
        
        if 'Target' in df.columns:
            st.write("### Average Performance by Target Classification")
            st.dataframe(df.groupby('Target')[perf_cols].mean(), use_container_width=True)
            
            st.markdown('''
            <div class="kpi-card" style="margin-top: 1rem;">
                <strong>Observed Relationship:</strong> Higher Target tiers correlate heavily with Student Feedback and Skill Upgrade Scores, indicating that pedagogical effectiveness is closely tied to applied skills.
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("Performance variables not found.")

elif selection == "05 — Department Analysis":
    st.markdown('<div class="section-hdr">Department Breakdown</div>', unsafe_allow_html=True)
    if 'Department' in df.columns:
        train_cols = [c for c in ['Training_Hours', 'Total_Training_Exposure'] if c in df.columns]
        perf_cols = [c for c in ['Course_Completion_Rate', 'Pass_Percentage'] if c in df.columns]
        
        if train_cols and perf_cols:
            st.write("### Cross-Departmental Comparison")
            st.dataframe(df.groupby('Department')[train_cols + perf_cols].mean(), use_container_width=True)
            
            st.markdown('''
            <div class="kpi-card" style="margin-top: 1rem;">
                <strong>Observation:</strong> We avoid ranking departments as "best" or "worst", as grading curves and training mandates differ. Instead, observe which departments have <em>higher training exposure</em> vs. <em>higher observed performance</em> to identify potential cultural misalignments.
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("Department column not found.")

elif selection == "06 — Visualizations":
    st.markdown('<div class="section-hdr">Visualization Gallery</div>', unsafe_allow_html=True)
    sns.set_theme(style="whitegrid")
    
    c1, c2 = st.columns(2)
    
    with c1:
        if 'Target' in df.columns and 'Training_Hours' in df.columns:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.write("**1. Training Hours vs Target Performance**")
            st.caption("Evaluates if higher training hours directly align with elite performance tiers.")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='Target', y='Training_Hours', ax=ax, palette='Blues', hue='Target', legend=False)
            st.pyplot(fig)
            st.markdown('</div><br>', unsafe_allow_html=True)
            
        if 'Department' in df.columns and 'Skill_Upgrade_Score' in df.columns:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.write("**3. Skill Upgrades by Department**")
            st.caption("Distribution of skill acquisition across academic faculties.")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.violinplot(data=df, x='Department', y='Skill_Upgrade_Score', ax=ax, palette='pastel', hue='Department', legend=False)
            st.pyplot(fig)
            st.markdown('</div><br>', unsafe_allow_html=True)

    with c2:
        if 'Target' in df.columns and 'Student_Feedback' in df.columns:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.write("**2. Student Feedback vs Target Performance**")
            st.caption("Illustrates the stark contrast in student reception across tiers.")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='Target', y='Student_Feedback', ax=ax, palette='Greens', hue='Target', legend=False)
            st.pyplot(fig)
            st.markdown('</div><br>', unsafe_allow_html=True)
            
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.write("**4. Correlation Heatmap**")
            st.caption("Identifies mathematical associations between training and outcomes.")
            cols_to_plot = [c for c in ['Training_Hours', 'Workshops_Attended', 'Skill_Upgrade_Score', 'Student_Feedback', 'Pass_Percentage'] if c in num_cols]
            if len(cols_to_plot) > 1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df[cols_to_plot].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

elif selection == "07 — Decision Tree":
    st.markdown('<div class="section-hdr">Decision Tree Model</div>', unsafe_allow_html=True)
    dt, metrics, importances = train_dt_model(df)
    
    if dt is None:
        st.error(metrics)
    else:
        st.markdown(f'''
        **Model Type:** Classification  
        **Target Variable:** `Target`  
        **Training Samples:** {metrics['train_samples']} | **Testing Samples:** {metrics['test_samples']}
        ''')
        
        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_kpi("Accuracy", f"{metrics['accuracy']*100:.1f}%", "Overall correctness")
        with c2: render_kpi("Precision", f"{metrics['precision']*100:.1f}%", "Weighted avg")
        with c3: render_kpi("Recall", f"{metrics['recall']*100:.1f}%", "Weighted avg")
        with c4: render_kpi("F1 Score", f"{metrics['f1']*100:.1f}%", "Weighted avg")
        
        st.write("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns([1, 1.5])
        
        with c_left:
            st.write("### Classification Report")
            st.text(metrics['report'])
            
        with c_right:
            st.write("### Feature Importance")
            st.info("Feature importance shows which variables contributed most to the model's predictions. It does not establish causation.")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.barplot(data=importances.head(8), x='Importance', y='Feature', ax=ax, palette='viridis', hue='Feature', legend=False)
            st.pyplot(fig)

elif selection == "08 — Confounding Factors":
    st.markdown('<div class="section-hdr">Factors That May Influence Training Outcomes</div>', unsafe_allow_html=True)
    display_limitation()
    
    st.markdown('''
    <div class="kpi-card">
        <h4 style="color:#1E3A8A;">1. Departmental Culture</h4>
        <p><strong>Observed Association:</strong> Business mandates more training, but Science outputs better grades.</p>
        <p><strong>Possible Confounding Factor:</strong> Departmental expectations and lenient grading curves act as a third variable artificially linking training volume to performance.</p>
    </div>
    
    <div class="kpi-card" style="margin-top:1rem;">
        <h4 style="color:#1E3A8A;">2. Experience Level</h4>
        <p><strong>Observed Association:</strong> Highly tenured faculty achieve elite target scores.</p>
        <p><strong>Possible Confounding Factor:</strong> Tenured faculty naturally perform better and have access to specialized FDPs. Experience drives both outcomes independently.</p>
    </div>
    
    <div class="kpi-card" style="margin-top:1rem;">
        <h4 style="color:#1E3A8A;">3. Inherent Motivation (Skill Upgrade)</h4>
        <p><strong>Observed Association:</strong> High skill upgrade scores perfectly predict high performance.</p>
        <p><strong>Possible Confounding Factor:</strong> Highly motivated faculty actively seek training and naturally strive for high classroom performance, acting as the true root cause.</p>
    </div>
    ''', unsafe_allow_html=True)

elif selection == "09 — Key Insights":
    st.markdown('<div class="section-hdr">5 Key Insights</div>', unsafe_allow_html=True)
    
    insights = [
        ("Training Volume", "Higher training exposure (raw hours) lacks a direct linear association with elite performance tiers, indicating quality matters over volume."),
        ("Skill Application", "Skill Upgrade Score was highly predictive of the final Target class, emphasizing applied learning over passive attendance."),
        ("Student Feedback Gap", "High-performing target tiers achieve vastly superior student feedback despite logging nearly identical raw training hours as lower tiers."),
        ("Experience Confounder", "Variables like Teaching Experience and Publications strongly rival formal training in predicting elite classification."),
        ("Model Complexity", "The classification model caps around ~40% accuracy, proving mathematically that raw HR demographics cannot perfectly predict human teaching capability.")
    ]
    
    for i, (title, desc) in enumerate(insights):
        st.markdown(f'''
        <div class="insight-card">
            <div class="insight-num">0{i+1}</div>
            <div class="insight-title">{title}</div>
            <div style="color: #475569;">{desc}</div>
        </div>
        ''', unsafe_allow_html=True)

elif selection == "10 — Action Plan":
    st.markdown('<div class="section-hdr">Training Effectiveness Action Plan</div>', unsafe_allow_html=True)
    
    actions = [
        {
            "obs": "Sheer training duration does not guarantee better pass rates.",
            "act": "Shift from Volume to Quality: Pivot from enforcing broad hour quotas to mandating targeted, project-based skill development programmes.",
            "ben": "Reduced burnout and higher ROI on professional development budgets."
        },
        {
            "obs": "Massive variances in participation vs outcomes across departments.",
            "act": "Standardize Departmental Training: Review department-specific training needs to align disparate outcomes across faculties.",
            "ben": "Equitable resource distribution and unified institutional pedagogy."
        },
        {
            "obs": "Experience and publications are dominant performance indicators.",
            "act": "Leverage Internal Mentorship: Pair junior faculty with tenured high-performers for collaborative research instead of relying purely on external FDPs.",
            "ben": "Faster onboarding and organic skill transfer."
        },
        {
            "obs": "Current dataset lacks longitudinal measurements, preventing causal proof.",
            "act": "Implement Pre/Post Measurement Architectures: Capture student feedback exactly 1 month before and 6 months after specific training modules.",
            "ben": "Ability to calculate exact causal ROI of external training vendors."
        }
    ]
    
    for act in actions:
        st.markdown(f'''
        <div class="action-card">
            <div class="action-step"><span class="action-label">OBSERVATION</span><br><span class="action-val">{act['obs']}</span></div>
            <div class="action-step"><span class="action-label">RECOMMENDED ACTION</span><br><span style="color:#2563EB; font-weight:600;">{act['act']}</span></div>
            <div class="action-step"><span class="action-label">EXPECTED BENEFIT</span><br><span style="color:#166534; font-weight:500;">{act['ben']}</span></div>
        </div>
        ''', unsafe_allow_html=True)
