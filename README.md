# 🌿 AgroAssist — AI/ML Banana Farm Intelligence System

> AI-powered surrogate NPK estimation and early disease prediction for banana cultivation — based on published VIT patent (IDF-B, TRL 7)

## 🔬 What This Project Does

AgroAssist estimates soil Nitrogen, Phosphorus and Potassium levels **without lab tests** using cheap IoT sensor readings — the surrogate concept from my published patent. It also predicts Sigatoka fungal disease and Fusarium wilt risk before visible symptoms appear.

## 🤖 ML Models

| Model | Accuracy | Purpose |
|-------|----------|---------|
| Stress Classifier | 100% | Detects healthy / N-deficient / Sigatoka / water stress |
| NPK — Nitrogen | 94.1% R² | Estimates N level surrogately |
| NPK — Phosphorus | 60.6% R² | Estimates P level surrogately |
| NPK — Potassium | 69.5% R² | Estimates K level surrogately |
| Sigatoka Risk | 98.4% R² | Predicts fungal disease probability |
| Fusarium Risk | 94.3% R² | Predicts soil-borne disease probability |

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 🛠️ Tech Stack

Python · Scikit-learn · Random Forest · XGBoost · Streamlit · Pandas · NumPy · Plotly · Joblib

## 📁 Project Structure

    agroassist/
    ├── notebooks/     # Phase 1: Dataset simulation, Phase 2: Model training
    ├── data/          # Generated sensor dataset + EDA plots
    ├── models/        # Trained ML models (.pkl files)
    └── app/           # Streamlit dashboard