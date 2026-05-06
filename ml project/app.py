import streamlit as st
import pandas as pd
from model import load_data, train_model, prepare_single_input

def main():
    st.set_page_config(page_title="Employee Attrition Risk Scorer", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Theme & Colors */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #F5F3EE;
            color: #111111;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stHeader"] {
            background-color: transparent;
        }

        section[data-testid="stSidebar"] {
            background-color: #F8F6F2 !important;
            border-right: 1px solid #E5E2DC;
        }

        .stApp {
            color: #111111;
        }

        /* Typography overrides */
        p, span, div {
            color: #111111;
        }
        
        .css-1d391kg, .css-1lcbmhc, .css-5tx12u { color: #111111; }
        .css-1u0zb7k, .css-1offfwp, .css-1n76uvr { color: #6B6B6B; }

        .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }

        /* Page heading */
        .page-heading {
            padding: 1rem 0 1.5rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid #111111;
            display: flex;
            align-items: baseline;
            justify-content: space-between;
        }

        .page-heading h1 {
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #111111;
        }

        .page-heading p {
            margin: 0;
            color: #6B6B6B;
            font-size: 1.1rem;
            font-weight: 600;
            background: #FFFFFF;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            border: 1px solid #E5E2DC;
        }

        /* Cards and panels */
        .summary-card,
        .glass-panel,
        .info-card {
            border-radius: 24px;
            background: #FFFFFF;
            border: 1px solid #E5E2DC;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
            margin-bottom: 1.2rem;
        }

        .summary-card {
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .summary-card-row {
            display: flex;
            align-items: flex-start;
            gap: 1.5rem;
        }

        .summary-card h3,
        .info-card h3,
        .glass-panel h2 {
            margin: 0 0 0.5rem;
            color: #111111;
            font-weight: 700;
            font-size: 1.2rem;
        }

        .summary-card .value {
            margin: 0;
            font-size: 4.5rem;
            font-weight: 800;
            color: #111111;
            line-height: 0.9;
            letter-spacing: -0.04em;
        }

        .summary-card .subtext,
        .glass-panel p,
        .info-card p {
            margin: 0.5rem 0 0;
            color: #6B6B6B;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .glass-panel {
            padding: 1.5rem;
        }
        
        .info-card {
            padding: 1.5rem;
            background: #F8F6F2;
        }

        /* Pills */
        .info-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            color: #111111;
            background: #FFFFFF;
            border: 1px solid #E5E2DC;
        }

        .info-pill.green { border-left: 4px solid #10b981; }
        .info-pill.amber { border-left: 4px solid #FFD84D; background: #FFD84D; border-color: #FFD84D; }
        .info-pill.red { border-left: 4px solid #ef4444; }

        /* Buttons */
        .stButton>button,
        .stDownloadButton>button {
            border-radius: 999px !important;
            padding: 0.5rem 1.2rem !important;
            font-weight: 600 !important;
            background: #FFFFFF !important;
            color: #111111 !important;
            border: 1px solid #E5E2DC !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
            transition: all 0.2s ease !important;
        }

        .stButton>button[kind="primary"] {
            background: #FFD84D !important;
            color: #111111 !important;
            border-color: #FFD84D !important;
            box-shadow: 0 4px 12px rgba(255, 216, 77, 0.2) !important;
        }

        .stButton>button:hover,
        .stDownloadButton>button:hover {
            border-color: #111111 !important;
        }
        
        .stButton>button[kind="primary"]:hover {
            background: #ffcf2a !important;
            border-color: #111111 !important;
        }

        /* Inputs */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div {
            border-radius: 12px !important;
            border: 1px solid #E5E2DC !important;
            background-color: #FFFFFF !important;
        }
        
        div[data-baseweb="input"]:focus-within > div, 
        div[data-baseweb="select"] > div:focus-within {
            border-color: #FFD84D !important;
            box-shadow: 0 0 0 1px #FFD84D !important;
        }
        
        .stNumberInput input {
            color: #111111 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 3rem !important;
            font-weight: 800 !important;
            color: #111111 !important;
            letter-spacing: -0.02em;
        }

        [data-testid="stMetricLabel"] {
            font-weight: 600 !important;
            color: #6B6B6B !important;
        }

        /* Risk Levels */
        .risk-high {
            color: #ef4444;
            font-weight: 800;
            margin: 0;
            font-size: 1.5rem;
        }

        .risk-medium {
            color: #f59e0b;
            font-weight: 800;
            margin: 0;
            font-size: 1.5rem;
        }

        .risk-low {
            color: #10b981;
            font-weight: 800;
            margin: 0;
            font-size: 1.5rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0 !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 0 !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #111111 !important;
            border-bottom-color: #FFD84D !important;
            font-weight: 700 !important;
        }

        .stTabs [role="tabpanel"] {
            background: transparent !important;
            border: none !important;
            padding: 1.5rem 0 !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #FFD84D;
        }
        
        /* Sidebar Text */
        [data-testid="stSidebar"] * {
            color: #111111 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] .stMarkdown li {
            color: #6B6B6B !important;
        }
        [data-testid="stSidebar"] h2 {
            color: #111111 !important;
            font-weight: 700 !important;
        }
        
        /* Checkbox inside sidebar/inputs */
        [data-baseweb="checkbox"] div {
            color: #111111 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-heading">
      <h1>Employee Risk Scorer</h1>
      <p>AI Assistant / Share</p>
    </div>
    """, unsafe_allow_html=True)

    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None

    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = None

    # Sidebar: Model Management
    with st.sidebar:
        st.markdown("## Model Configuration")
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

        st.markdown("""
        **About this model:**
        - Algorithm: Random Forest (200 trees)
        - Training: 80% data, tested on 20%
        - Predicts attrition probability (0-100%)
        """)

    count = df.shape[0]
    attrition_pct = None
    if 'Attrition' in df.columns:
        attrition_pct = (df['Attrition'].astype(str).str.strip().eq('Yes').mean() * 100)

    # Main content layout
    top_left, top_center, top_right = st.columns([1.1, 2.2, 1.0])

    with top_left:
        st.markdown('<div class="summary-card" style="min-height: 160px;"><div class="summary-card-row"><p class="value">{}</p><div><h3>Team Size</h3><p class="subtext">Total employees</p></div></div></div>'.format(count), unsafe_allow_html=True)
        if attrition_pct is not None:
            st.markdown('<div class="summary-card" style="min-height: 160px;"><div class="summary-card-row"><p class="value">{:.1f}%</p><div><h3>Attrition Rate</h3><p class="subtext">Employees leaving</p></div></div></div>'.format(attrition_pct), unsafe_allow_html=True)
        else:
            st.markdown('<div class="summary-card" style="min-height: 160px;"><div class="summary-card-row"><p class="value">N/A</p><div><h3>Attrition Rate</h3><p class="subtext">No labels</p></div></div></div>', unsafe_allow_html=True)

    with top_center:
        st.markdown('<div class="glass-panel" style="min-height: 345px;"><h2>Active HR Dashboard</h2><p>Use this workspace to score attrition risk, review employee trends, and run batch predictions in one place.</p><div style="margin-top: 1.5rem; border-left: 2px solid #FFD84D; padding-left: 1rem;"><h3 style="margin:0; font-size: 1.05rem;">Latest Activity</h3><p style="margin: 0.2rem 0 0; color: #6B6B6B; font-size: 0.9rem;">Model trained with {} employees. Ready for predictions.</p></div><div style="margin-top: 1rem; border-left: 2px solid #E5E2DC; padding-left: 1rem;"><h3 style="margin:0; font-size: 1.05rem;">System Status</h3><p style="margin: 0.2rem 0 0; color: #6B6B6B; font-size: 0.9rem;">All services running optimally.</p></div></div>'.format(count), unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="info-card" style="min-height: 345px;"><h3>Ai Assistant</h3><p class="info-pill green">Live predictions</p><p class="info-pill amber">Model tuning</p><p class="info-pill red">Retention alerts</p><p style="margin-top:1rem;color:#6B6B6B;font-size:0.9rem;line-height:1.6;">Review the model on the sidebar and update with fresh datasets anytime.</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])

    with tab1:
        left_col, right_col = st.columns([1.3, 1])

        with left_col:
            st.markdown('<div class="glass-panel"><h2>Employee risk input</h2><p>Fill the employee details and run the risk score. The interface is styled to feel like a modern productivity app.</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=16, max_value=100, value=30, help="Employee age in years")
            monthly_income = st.number_input("Monthly Income", min_value=0, value=3000, help="Monthly salary in USD")
            job_level = st.number_input("Job Level", min_value=0, max_value=5, value=1, help="Career level (1=Entry to 5=Executive)")
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=2, help="Tenure in years")
            overtime = st.selectbox("Overtime", options=["No", "Yes"], help="Works overtime regularly?") if 'Overtime' in df.columns else "No"
            distance = st.number_input("Distance From Home", min_value=0, max_value=100, value=5, help="Commute distance in miles") if 'Distance_From_Home' in df.columns else 0
            st.markdown('</div>', unsafe_allow_html=True)

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

        with right_col:
            st.markdown('<div class="glass-panel"><h2>Risk insights</h2><p>Results are displayed instantly with a clean risk summary and progress visualization.</p></div>', unsafe_allow_html=True)
            if st.session_state['last_prediction'] is not None:
                proba = st.session_state['last_prediction']
                st.metric("Attrition Probability", f"{proba:.1%}")
                if proba >= 0.6:
                    st.markdown('<p class="risk-high">HIGH RISK</p>', unsafe_allow_html=True)
                    st.warning("This employee has a **high** probability of leaving. Consider retention strategies.")
                elif proba >= 0.4:
                    st.markdown('<p class="risk-medium">MEDIUM RISK</p>', unsafe_allow_html=True)
                    st.info("This employee shows moderate attrition risk. Monitor engagement.")
                else:
                    st.markdown('<p class="risk-low">LOW RISK</p>', unsafe_allow_html=True)
                    st.success("This employee is likely to stay.")
                st.progress(proba, text=f"Risk score: {proba:.1%}")
            else:
                st.info("Predict the risk score to review the results here.")

    with tab2:
        st.markdown('<div class="glass-panel"><h2>Batch Analysis</h2><p>Upload a CSV file to run attrition scoring for multiple employees. Download the results after processing.</p></div>', unsafe_allow_html=True)

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
                        results_df['Risk_Level'] = results_df['Attrition_Probability'].apply(lambda x: 'HIGH' if x >= 0.6 else ('MEDIUM' if x >= 0.4 else 'LOW'))
                        st.session_state['batch_results'] = results_df
                except Exception as e:
                    st.error(f"Error during batch analysis: {str(e)}")

        if st.session_state['batch_results'] is not None:
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
            st.download_button("Download Predictions (CSV)", csv, "attrition_predictions.csv", "text/csv", use_container_width=True)

    st.divider()
    st.caption("**Employee Attrition Risk Scorer** | Built with Random Forest ML | For HR decision support only")

if __name__ == '__main__':
    main()
