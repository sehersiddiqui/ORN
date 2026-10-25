# ORN Research Progress Dashboard

A local Streamlit + Plotly dashboard for the ORN research progress review.

## Quick start (Windows)
1. Install Python 3.10–3.12.
2. Extract this folder.
3. Double-click `run_dashboard.bat`, or run these commands in PowerShell from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL Streamlit prints, usually `http://localhost:8501`.

## macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features
- Plain-language cohort overview and ORN prevalence
- Patient and tumour subgroup charts
- Interactive dose-volume plots
- Missingness, correlation and data audit
- Trainable logistic regression, Random Forest, Extra Trees, and optional XGBoost baselines
- ROC and precision–recall curves and confusion matrix
- Project-stage status and hospital data requirements
- Searchable data dictionary

## Data and research integrity
The app uses `TTE_ORN_data_Figshare.csv` bundled in this folder. It does not contain patient-linked serial CT radiomics or biomarker measurements. It therefore does not claim to train the proposed T0–T4 LSTM, ResNet-18 on ORN-labelled CT crops, or full multimodal Temporal Fusion Transformer.

Model metrics are exploratory results from a single holdout split. They are not external validation, and this dashboard is not a clinical decision-support tool. Do not use it for diagnosis or treatment decisions.
