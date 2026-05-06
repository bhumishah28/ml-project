import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

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
