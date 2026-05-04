# Employee Attrition Streamlit App

Run the Streamlit interface for the Employee Attrition project.

Quick start

1. Create and activate a Python environment (optional but recommended).

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

2. The app will try to find a dataset file in the project folder matching `employee_attrition` in its filename. If not found, upload your CSV via the UI.

3. Use the minimal-input form to get an attrition probability score.

Notes
- The app trains a RandomForest on first load and caches the model. Use the sidebar button to retrain.
- For best predictions, upload full employee records matching the original dataset's column names.
