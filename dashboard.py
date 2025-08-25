# app_dash.py  — Rebuilt Step 18 with Dash (Python 3.9)

import os
import io
import json
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import joblib

from dash import Dash, dcc, html, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate

# -----------------------------
# Config
# -----------------------------
ARTIFACTS_DIR = "artifacts"
DATA_PATH = "/mnt/data/ncr_ride_bookings.csv"  # optional default demo path

app = Dash(__name__)
app.title = "Uber Ride Cancellation Risk – Dash Dashboard"

# -----------------------------
# Utilities & ML glue (match training script)
# -----------------------------

# Re-declare TopKCategorical so joblib can unpickle the saved preprocess pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

class TopKCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, top_k: int = 30):
        self.top_k = top_k
        self.keep_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.keep_maps_ = {}
        for col in X.columns:
            vc = Counter(X[col].astype(str).fillna("nan"))
            keep = set([c for c, _ in vc.most_common(self.top_k)])
            self.keep_maps_[col] = keep
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            keep = self.keep_maps_.get(col, set())
            X[col] = X[col].astype(str).where(X[col].astype(str).isin(keep), other="Other")
        return X

def add_booking_time_features(df: pd.DataFrame) -> pd.DataFrame:
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
FEATURE_COLS = [c for c in NUM_COLS + CAT_COLS]

def load_artifacts_or_none():
    try:
        prep = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocess.joblib"))
        booster = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.joblib"))
        with open(os.path.join(ARTIFACTS_DIR, "risk_thresholds.json"), "r", encoding="utf-8") as f:
            thr = json.load(f)
        return prep, booster, thr, ""
    except Exception as e:
        msg = f"Artifacts missing or unreadable in '{ARTIFACTS_DIR}'. Run training (Steps 8–15) first. Details: {e}"
        return None, None, None, msg

def predict_proba(df_in: pd.DataFrame, prep, booster) -> np.ndarray:
    dfx = add_booking_time_features(df_in)
    # Align expected columns
    missing = [c for c in FEATURE_COLS if c not in dfx.columns]
    for m in missing:
        dfx[m] = np.nan
    X = dfx[FEATURE_COLS]
    Xenc = prep.transform(X)
    p = booster.predict_proba(Xenc)[:, 1]
    return p

def band_one(p: float, thr: dict) -> str:
    return "High" if p >= thr["t_cut"] else "Low"


# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div([
    html.H2("Uber Ride Cancellation Risk – Dash Dashboard"),

    html.Div([
        html.Div([
            html.P("Upload Excel/CSV with upcoming bookings:"),
            dcc.Upload(
                id="file-upload",
                children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                    "textAlign": "center", "margin": "10px"
                },
                multiple=False
            ),
            html.Div(id="artifact-warning", style={"color": "crimson", "marginTop": "5px"}),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "20px"}),

    dcc.Store(id="stored-data"),  # stores scored dataframe as JSON

    html.Hr(),

    html.Div([
        html.H4("Scored Orders (first 200 rows)"),
        dash_table.DataTable(
            id="table",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px", "whiteSpace": "normal"},
        ),
        html.Br(),
        html.Button("Download Scored CSV (UTF-8)", id="download-btn"),
        dcc.Download(id="download-csv"),
    ]),

    html.Hr(),

    html.Div([
        html.Div([
            html.H4("Risk Probability Distribution"),
            dcc.Graph(id="hist-prob")
        ], style={"flex": "1"}),
        html.Div([
            html.H4("Share by Risk Band"),
            dcc.Graph(id="share-band")
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "20px"}),
], style={"padding": "20px"})

# -----------------------------
# Callbacks
# -----------------------------

def parse_contents(contents, filename):
    """
    Decode uploaded file content to DataFrame.
    CSV assumed UTF-8; Excel handled by pandas engine.
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(decoded))
    else:
        return pd.read_csv(io.StringIO(decoded.decode("utf-8")), encoding="utf-8")

@app.callback(
    Output("stored-data", "data"),
    Output("artifact-warning", "children"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        raise PreventUpdate
    df_in = parse_contents(contents, filename)
    prep, booster, thr, warn = load_artifacts_or_none()
    if prep is None:
        # Return the raw uploaded data but warn; no scoring possible
        return df_in.to_json(date_format="iso", orient="split"), warn

    # Score
    p = predict_proba(df_in, prep, booster)
    df_out = df_in.copy()
    df_out["CancelRisk_Prob"] = p
    df_out["CancelRisk_Band"] = [band_one(v, thr) for v in p]
    return df_out.to_json(date_format="iso", orient="split"), ""

@app.callback(
    Output("table", "data"),
    Output("table", "columns"),
    Output("hist-prob", "figure"),
    Output("share-band", "figure"),
    Input("stored-data", "data"),
    prevent_initial_call=True
)
def refresh_views(data_json):
    if data_json is None:
        raise PreventUpdate
    df = pd.read_json(data_json, orient="split")
    # DataTable
    cols = [{"name": c, "id": c} for c in df.columns]
    data = df.head(200).to_dict("records")

    # Histogram
    if "CancelRisk_Prob" in df.columns:
        fig_hist = px.histogram(df, x="CancelRisk_Prob", nbins=30, color="CancelRisk_Band",
                                title="Cancellation Risk Probability")
    else:
        fig_hist = px.histogram(title="Upload and score to see probabilities")

    # Share by band
    if "CancelRisk_Band" in df.columns:
        order = ["Low", "High"]
        share = (df["CancelRisk_Band"]
                 .value_counts(normalize=True)
                 .rename_axis("band").reset_index(name="share"))
        share["band"] = pd.Categorical(share["band"], categories=order, ordered=True)
        share = share.sort_values("band")
        fig_share = px.bar(share, x="band", y="share", text="share", title="Share by Risk Band")

    else:
        fig_share = px.bar(title="Upload and score to see bands")

    return data, cols, fig_hist, fig_share

@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def download_scored(n, data_json):
    if not n:
        raise PreventUpdate
    if data_json is None:
        raise PreventUpdate
    df = pd.read_json(data_json, orient="split")
    # Use dcc.send_data_frame helper (keeps UTF-8)
    return dcc.send_data_frame(df.to_csv, "scored_orders.csv", index=False, encoding="utf-8")

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # Helpful message if artifacts are missing
    if not os.path.isdir(ARTIFACTS_DIR):
        print(f"Note: '{ARTIFACTS_DIR}' directory not found. Create it and run training (Steps 8–15).")
    else:
        expected = ["preprocess.joblib","xgb_model.joblib","risk_thresholds.json"]
        missing = [f for f in expected if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))]
        if missing:
            print(f"Warning: Missing artifacts: {missing}. Run training (Steps 8–15) to enable scoring.")
    app.run(debug=True, host="127.0.0.1", port=8050)

