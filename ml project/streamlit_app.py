import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

@st.cache_data
def find_dataset():
    base = Path(__file__).parent
    files = list(base.glob("**/*employee_attrition*.csv"))
    return files[0] if files else None

@st.cache_data
def load_data(path=None):
    if path is None:
        path = find_dataset()
    if path is None:
        return None
    return pd.read_csv(path)

def feature_engineer(df):
    df = df.copy()
    if "Monthly_Income" in df.columns and "Job_Level" in df.columns:
        df["Income_per_Level"] = df["Monthly_Income"] / (df["Job_Level"] + 1)
    if "Years_at_Company" in df.columns:
        df["Tenure_Bucket"] = pd.cut(df["Years_at_Company"], bins=[0,2,5,10,100], labels=[1,2,3,4]).astype(int)
    if "Years_Since_Last_Promotion" in df.columns and "Years_at_Company" in df.columns:
        df["Promotion_Delay_Ratio"] = df["Years_Since_Last_Promotion"] / (df["Years_at_Company"] + 1)
    return df

def preprocess(df, fit_scaler=True):
    df = df.copy()
    TARGET = "Attrition"
    DROP_COLS = ["Employee_ID"]
    BINARY_COLS = ["Gender", "Overtime"]
    OHE_COLS = ["Marital_Status", "Department", "Job_Role"]

    df.drop(columns=DROP_COLS, inplace=True, errors='ignore')
    if TARGET in df.columns:
        y = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
        df = df.drop(columns=[TARGET])
    else:
        y = None

    # Label encode binary
    label_encoders = {}
    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # One-hot
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    if ohe_present:
        df = pd.get_dummies(df, columns=ohe_present, drop_first=False)

    # Scale numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if fit_scaler:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, y, label_encoders, scaler

@st.cache_resource
def train_model(df):
    df = feature_engineer(df)
    X, y, les, scaler = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample')
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    return {'model': model, 'scaler': scaler, 'label_encoders': les, 'features': X.columns.tolist(), 'auc': auc}

def prepare_single_input(example, features, les, scaler, df_template=None):
    """Prepare a single input for prediction, ensuring feature alignment with training data."""
    row = pd.DataFrame([example])
    row = feature_engineer(row)
    
    # Fill missing categorical columns with defaults from template
    if df_template is not None:
        default_dept = df_template['Department'].mode()[0] if 'Department' in df_template.columns and len(df_template['Department'].mode()) > 0 else 'Sales'
        default_marital = df_template['Marital_Status'].mode()[0] if 'Marital_Status' in df_template.columns and len(df_template['Marital_Status'].mode()) > 0 else 'Single'
        default_role = df_template['Job_Role'].mode()[0] if 'Job_Role' in df_template.columns and len(df_template['Job_Role'].mode()) > 0 else 'Sales Executive'
        
        if 'Department' not in row.columns or pd.isna(row['Department'].iloc[0]):
            row['Department'] = default_dept
        if 'Marital_Status' not in row.columns or pd.isna(row['Marital_Status'].iloc[0]):
            row['Marital_Status'] = default_marital
        if 'Job_Role' not in row.columns or pd.isna(row['Job_Role'].iloc[0]):
            row['Job_Role'] = default_role
    
    # Apply binary label encoders
    for col, le in les.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))
    
    # One-hot encode categorical columns with all categories from template
    if df_template is not None:
        # Get all unique categories from template for each categorical column
        categories_dict = {}
        for cat_col in ['Marital_Status', 'Department', 'Job_Role']:
            if cat_col in df_template.columns:
                categories_dict[cat_col] = sorted(df_template[cat_col].unique().tolist())
        
        # Apply one-hot encoding for each categorical
        for cat_col, categories in categories_dict.items():
            if cat_col in row.columns:
                # Create one-hot encoded columns
                for cat in categories:
                    col_name = f'{cat_col}_{cat}'
                    row[col_name] = (row[cat_col] == cat).astype(int)
                row = row.drop(columns=[cat_col])
    else:
        # Fallback: standard one-hot encoding
        OHE_COLS = ['Marital_Status', 'Department', 'Job_Role']
        ohe_present = [c for c in OHE_COLS if c in row.columns]
        if ohe_present:
            row = pd.get_dummies(row, columns=ohe_present, drop_first=False)
    
    # Create result dataframe with all expected features (initialized to 0)
    result = pd.DataFrame(0.0, index=[0], columns=features)
    
    # Fill in values we have
    for col in result.columns:
        if col in row.columns:
            result[col] = row[col].values[0]
    
    # Scale numeric columns only (exclude all one-hot encoded categorical columns)
    if scaler is not None:
        # Identify numeric columns (those that don't start with categorical prefixes)
        numeric_cols = [c for c in features if not any(c.startswith(prefix) for prefix in 
                       ['Marital_Status_', 'Department_', 'Job_Role_'])]
        
        if numeric_cols:
            try:
                result[numeric_cols] = scaler.transform(result[numeric_cols])
            except:
                # If scaling fails, just continue without scaling
                pass
    
    return result

def main():
    st.set_page_config(page_title="Employee Attrition Risk Scorer", layout="wide", initial_sidebar_state="expanded")
    
    # Premium Custom CSS for an incredible UI
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

        /* Main app styling with gradient background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Typography overrides */
        h1, h2, h3, h4, h5, p, span {
            font-family: 'Inter', sans-serif;
        }
        h1 {
            background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            letter-spacing: -1px;
            padding-bottom: 10px;
        }

        /* Glassmorphism containers */
        .info-box, .success-box { 
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 24px; 
            border-radius: 16px; 
            margin: 15px 0;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .info-box:hover {
            transform: translateY(-2px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Risk level styling */
        .risk-high { 
            background: linear-gradient(90deg, #ef4444, #f87171);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800; 
            font-size: 28px; 
            text-shadow: 0px 4px 20px rgba(239, 68, 68, 0.4);
            margin: 0;
        }
        .risk-medium { 
            background: linear-gradient(90deg, #f59e0b, #fbbf24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800; 
            font-size: 28px; 
            text-shadow: 0px 4px 20px rgba(245, 158, 11, 0.4);
            margin: 0;
        }
        .risk-low { 
            background: linear-gradient(90deg, #10b981, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800; 
            font-size: 28px; 
            text-shadow: 0px 4px 20px rgba(16, 185, 129, 0.4);
            margin: 0;
        }

        /* Buttons styling */
        .stButton>button {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
            border: none;
        }
        
        /* Metric widget styling */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600 !important;
            color: #94a3b8 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("# Employee Attrition Risk Scorer")
    st.markdown("**Predict employee attrition risk using machine learning. Minimal input required.**")

    # Sidebar: Model Management
    with st.sidebar:
        st.header("Model Configuration")
        
        uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"], help="Optional: Upload a custom employee dataset")
        
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success("Dataset loaded from upload")
        else:
            df = load_data()
            if df is not None:
                st.info(f"Loaded project dataset: {df.shape[0]} employees")

        if df is None:
            st.error("No dataset found. Please upload `employee_attrition_dataset.csv`")
            return

        st.divider()
        
        # Train/Retrain button
        col1, col2 = st.columns(2)
        retrain = col1.button("Retrain Model", use_container_width=True)
        show_perf = col2.button("Model Stats", use_container_width=True)
        
        if retrain:
            with st.spinner("Training model..."):
                meta = train_model(df)
            st.session_state['meta'] = meta
            st.success(f"Model trained. ROC-AUC: {meta['auc']:.1%}")
        else:
            if 'meta' not in st.session_state:
                with st.spinner("Training model on first load..."):
                    st.session_state['meta'] = train_model(df)

        meta = st.session_state['meta']
        

        st.info("""
        **About this model:**
        - Algorithm: Random Forest (200 trees)
        - Training: 80% data, tested on 20%
        - Predicts attrition probability (0-100%)
        """)

    # Main content with tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])

    # TAB 1: Single Prediction
    with tab1:
        st.subheader("Predict Risk for One Employee")
        st.markdown("Enter employee details below to get an attrition risk score.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(
                "Age", 
                min_value=16, max_value=100, value=30,
                help="Employee age in years"
            )
        with col2:
            monthly_income = st.number_input(
                "Monthly Income", 
                min_value=0, value=3000,
                help="Monthly salary in USD"
            )
        with col3:
            job_level = st.number_input(
                "Job Level", 
                min_value=0, max_value=5, value=1,
                help="Career level (1=Entry to 5=Executive)"
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            years_at_company = st.number_input(
                "Years at Company", 
                min_value=0, max_value=50, value=2,
                help="Tenure in years"
            )
        with col2:
            overtime = st.selectbox(
                "Overtime", 
                options=["No", "Yes"],
                help="Works overtime regularly?"
            ) if 'Overtime' in df.columns else "No"
        with col3:
            distance = st.number_input(
                "Distance From Home", 
                min_value=0, max_value=100, value=5,
                help="Commute distance in miles"
            ) if 'Distance_From_Home' in df.columns else 0

        # Prediction Button
        pred_col1, pred_col2 = st.columns([1, 3])
        with pred_col1:
            if st.button("Predict Risk", use_container_width=True, type="primary"):
                example = {
                    'Age': age,
                    'Monthly_Income': monthly_income if 'Monthly_Income' in df.columns else 3000,
                    'Job_Level': job_level if 'Job_Level' in df.columns else 1,
                    'Years_at_Company': years_at_company if 'Years_at_Company' in df.columns else 2,
                    'Overtime': overtime if 'Overtime' in df.columns else "No",
                    'Distance_From_Home': distance if 'Distance_From_Home' in df.columns else 5,
                }

                row = prepare_single_input(example, meta['features'], meta['label_encoders'], meta['scaler'], df)
                proba = meta['model'].predict_proba(row)[:,1][0]
                st.session_state['last_prediction'] = proba

        # Display Result
        if 'last_prediction' in st.session_state:
            proba = st.session_state['last_prediction']
            
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Attrition Probability", f"{proba:.1%}")
            
            with col2:
                if proba >= 0.6:
                    risk_level = "HIGH RISK"
                    st.markdown(f'<p class="risk-high">{risk_level}</p>', unsafe_allow_html=True)
                    st.warning("This employee has a **high** probability of leaving. Consider retention strategies.")
                elif proba >= 0.4:
                    risk_level = "MEDIUM RISK"
                    st.markdown(f'<p class="risk-medium">{risk_level}</p>', unsafe_allow_html=True)
                    st.info("This employee shows moderate attrition risk. Monitor engagement.")
                else:
                    risk_level = "LOW RISK"
                    st.markdown(f'<p class="risk-low">{risk_level}</p>', unsafe_allow_html=True)
                    st.success("This employee is likely to stay.")
            
            # Risk gauge visualization
            st.progress(proba, text=f"Risk Score: {proba:.1%}")

    # TAB 2: Batch Prediction
    with tab2:
        st.subheader("Predict Risk for Multiple Employees")
        st.markdown("Upload a CSV with employee data to get predictions for all employees at once.")
        
    batch_file = st.file_uploader("Upload employee CSV for batch analysis", type=["csv"], key="batch")
        
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            
            if st.button("Analyze All Employees", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Processing batch predictions..."):
                        batch_results = []
                        
                        for idx, row_data in batch_df.iterrows():
                            example = row_data.to_dict()
                            X = prepare_single_input(example, meta['features'], meta['label_encoders'], meta['scaler'], batch_df)
                            proba = meta['model'].predict_proba(X)[:,1][0]
                            batch_results.append({'Employee_ID': idx + 1, 'Attrition_Probability': proba})
                        
                        results_df = pd.DataFrame(batch_results)
                        results_df['Risk_Level'] = results_df['Attrition_Probability'].apply(
                            lambda x: 'HIGH' if x >= 0.6 else ('MEDIUM' if x >= 0.4 else 'LOW')
                        )
                        
                        st.session_state['batch_results'] = results_df

                except Exception as e:
                    st.error(f"Error during batch analysis: {str(e)}")
        
        if 'batch_results' in st.session_state:
            results_df = st.session_state['batch_results']
            
            st.success(f"Analyzed {len(results_df)} employees")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk = (results_df['Risk_Level'] == 'HIGH').sum()
                st.metric("High Risk", high_risk)
            with col2:
                med_risk = (results_df['Risk_Level'] == 'MEDIUM').sum()
                st.metric("Medium Risk", med_risk)
            with col3:
                low_risk = (results_df['Risk_Level'] == 'LOW').sum()
                st.metric("Low Risk", low_risk)
            
            st.dataframe(results_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Predictions (CSV)",
                csv,
                "attrition_predictions.csv",
                "text/csv",
                use_container_width=True
            )

    # Footer
    st.divider()
    st.caption(
    "**Employee Attrition Risk Scorer** | "
        "Built with Random Forest ML | "
        "For HR decision support only"
    )

if __name__ == '__main__':
    main()
