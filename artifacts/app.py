
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import streamlit as st

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "/mnt/data/ncr_ride_bookings.csv"  # change as needed

st.set_page_config(page_title="Uber Ride Cancellation Risk", layout="wide")

@st.cache_resource
def load_artifacts():
    prep = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocess.joblib"))
    booster = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "risk_thresholds.json")) as f:
        thr = json.load(f)
    return prep, booster, thr

def add_booking_time_features(df):
    import pandas as pd, numpy as np
    from datetime import datetime

    out = df.copy()
    out["is_cancelled"] = out["Booking Status"].astype(str).str.contains("cancel", case=False, na=False).astype(int)
    out["_Date_dt"] = pd.to_datetime(out["Date"], errors="coerce", infer_datetime_format=True)
    tparsed = pd.to_datetime(out["Time"], errors="coerce")
    out["_hour"] = tparsed.dt.hour
    out["is_weekend"] = out["_Date_dt"].dt.weekday.isin([5,6]).astype(int)
    out["is_late"] = out["_hour"].between(0,5, inclusive="both").astype(int)
    out["fare_per_km"] = np.where(out["Ride Distance"].fillna(0) > 0,
                                  out["Booking Value"] / out["Ride Distance"].replace(0, np.nan),
                                  np.nan)
    out["same_area"] = (out["Pickup Location"].astype(str).str.lower()
                        == out["Drop Location"].astype(str).str.lower()).astype(int)
    DROP = set([
        "Booking Status","Cancelled Rides by Customer","Reason for cancelling by Customer",
        "Cancelled Rides by Driver","Driver Cancellation Reason","Incomplete Rides",
        "Incomplete Rides Reason","Booking ID"
    ])
    out = out.drop(columns=[c for c in out.columns if c in DROP], errors="ignore")
    return out

NUM_COLS = ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance",
            "Driver Ratings","Customer Rating","fare_per_km","is_weekend","is_late","same_area","_hour"]
CAT_COLS = ["Customer ID","Vehicle Type","Pickup Location","Drop Location","Payment Method"]
FEATURE_COLS = NUM_COLS + CAT_COLS

st.title("Uber Ride Cancellation Risk ¨C Driver Plugin Prototype")

uploaded = st.file_uploader("Upload Excel/CSV with upcoming bookings", type=["xlsx","xls","csv"])
prep, booster, thr = load_artifacts()

def predict_proba(df_in):
    dfx = add_booking_time_features(df_in)
    X = dfx[FEATURE_COLS]
    Xenc = prep.transform(X)
    p = booster.predict_proba(Xenc)[:,1]
    return p

def band(p):
    if p >= thr["t_high"]:
        return "High"
    elif p >= thr["t_low"]:
        return "Medium"
    else:
        return "Low"

if uploaded is not None:
    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext in [".xlsx",".xls"]:
        df_in = pd.read_excel(uploaded)
    else:
        df_in = pd.read_csv(uploaded)

    p = predict_proba(df_in)
    df_out = df_in.copy()
    df_out["CancelRisk_Prob"] = p
    df_out["CancelRisk_Band"] = [band(v) for v in p]

    st.subheader("Scored Orders")
    st.dataframe(df_out.head(50))

    st.download_button("Download Scored File", df_out.to_csv(index=False).encode(), file_name="scored_orders.csv")

    st.subheader("Risk Distribution")
    fig = px.histogram(df_out, x="CancelRisk_Prob", nbins=30, color="CancelRisk_Band")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload an Excel/CSV file to score cancellation risk.")
