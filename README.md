# ML Project

An ML-based employee attrition risk scoring project with a Streamlit web app for fast, interactive predictions.

## Features
- End-to-end ML workflow: data loading, training, and inference
- Streamlit UI for interactive predictions
- Automatic model training and caching on first run
- Supports CSV upload when the dataset is not found locally

## Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Folder Structure
```
ml-project/
  README.md
  ml project/
    employee_attrition_dataset.csv
    requirements.txt
    streamlit_app.py
    README.md
```

## Prerequisites
- Python 3.9+

## Setup
1. Open a terminal in this repo and go to the app folder:
   ```powershell
   cd "ml project"
   ```
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Run the App
```powershell
streamlit run streamlit_app.py
```

## Usage
- The app searches for a dataset file containing `employee_attrition` in its name.
- If not found, upload the CSV in the UI.
- Fill the form to generate an attrition probability score.

## Future Improvements
- Add model evaluation metrics and reporting
- Introduce hyperparameter tuning and model versioning
- Expand feature engineering and data validation
- Add automated tests and CI pipeline
