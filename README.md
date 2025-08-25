# Uber RideRiskX

## Overview
Uber RideRiskX is a machine learning project that predicts the probability of ride cancellations for Uber bookings.  
Using the **2024 Uber Ride Analytics dataset (148,770 rides)**, we built a binary classification model with **XGBoost** that achieved **99% accuracy**.  
The project also includes an interactive **Dash dashboard** that allows users to upload Excel/CSV files and immediately obtain cancellation risk labels (Low / High) for each order.

---

## Features
- 🚖 **Predictive Model**: XGBoost trained on engineered features (time, distance, payment method, etc.).  
- 📊 **High Accuracy**: ~99% prediction accuracy on test data.  
- 🧹 **Preprocessing**: Scikit-learn pipelines with feature engineering (`is_weekend`, `is_late`, `fare_per_km`, `same_area`).  
- 📈 **Dashboard**: Interactive Dash web app for drag-and-drop Excel/CSV uploads.  
- 📂 **Artifacts**: Model (`xgb_model.joblib`), preprocessing pipeline, and thresholds stored for deployment.

---

## Tech Stack
- **Python 3.9**
- **XGBoost**
- **Scikit-learn**
- **Pandas / NumPy**
- **Plotly & Dash** (for dashboard)
- **Joblib** (artifact persistence)

---

## How It Works
1. **Data Preparation**  
   - Feature engineering on booking metadata (`Date`, `Time`, `Ride Distance`, etc.).  
   - Binary label: `1 = Cancelled`, `0 = Completed`.

2. **Model Training**  
   - Time-based train/validation/test split.  
   - XGBoost classifier with imbalance handling (`scale_pos_weight`).  
   - Probability calibration for better thresholding.

3. **Risk Scoring**  
   - Orders are labeled **High** risk if predicted probability ≥ threshold, otherwise **Low**.  
   - Threshold optimized on validation set with a cost-sensitive objective.

4. **Dashboard**  
   - Users drag-and-drop Excel/CSV files.  
   - Dashboard returns cancellation probabilities and risk bands, with visualizations (histograms and bar plots).  

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
