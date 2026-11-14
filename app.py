# app.py - Day 11: YouTube Viral Video Model-Tracking & Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Custom Transformer Definition (Required for Joblib deserialization)
class ViralFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_proc = X.copy()
        df_proc['log_likes'] = np.log1p(df_proc['likes'])
        df_proc['log_comments'] = np.log1p(df_proc['comments'])
        df_proc['log_subscribers'] = np.log1p(df_proc['subscriber_count'])

        df_proc['log_like_per_sub'] = df_proc['log_likes'] - df_proc['log_subscribers']
        df_proc['log_comment_per_sub'] = df_proc['log_comments'] - df_proc['log_subscribers']
        df_proc['log_engagement_sum'] = df_proc['log_likes'] + df_proc['log_comments']

        for col in self.feature_columns:
            if col not in df_proc.columns:
                df_proc[col] = 0
        return df_proc[self.feature_columns]

# 2. Page Configuration & Model Loading
st.set_page_config(page_title="YouTube Virality Predictor & Model Tracker", layout="wide")

@st.cache_resource
def load_production_artifact():
    return joblib.load("youtube_viral_champion_v1.joblib")

artifact = load_production_artifact()
pipeline = artifact['pipeline']
optimal_threshold = artifact['optimal_threshold']
metrics = artifact['metrics']

# 3. Header
st.title("YouTube Viral Video ML Tracker & Live Predictor")
st.markdown("End-to-end production deployment for the **12,000-Video YouTube Virality Dataset**.")

# 4. Tabs: Live Inference vs. Model Experiment Tracker
tab1, tab2 = st.tabs(["Live Video Virality Predictor", "Model Experiment Registry (Days 6–10)"])

# --- TAB 1: LIVE PREDICTION ---
with tab1:
    st.subheader("Simulate a New YouTube Video Upload")
    col1, col2, col3 = st.columns(3)

    with col1:
        likes = st.number_input("Likes Count", min_value=0, max_value=10_000_000, value=25000, step=500)
        comments = st.number_input("Comments Count", min_value=0, max_value=1_000_000, value=1200, step=50)
        subscribers = st.number_input("Channel Subscribers", min_value=1, max_value=100_000_000, value=50000, step=1000)

    with col2:
        title_length = st.slider("Title Length (Characters)", 5, 100, 52)
        tags_count = st.slider("Number of Tags", 0, 50, 15)
        duration_sec = st.slider("Video Duration (Seconds)", 15, 3600, 600)

    with col3:
        upload_hour = st.slider("Upload Hour (0–23)", 0, 23, 18)
        is_weekend = st.selectbox("Uploaded on Weekend?", options=[0, 1], format_func=lambda x: "Yes (Weekend)" if x == 1 else "No (Weekday)")
        category = st.selectbox("Video Category", ["Tech", "Gaming", "Vlog", "Education", "Music", "Comedy", "Entertainment"])

    if st.button("Predict Virality", type="primary"):
        # Build input DataFrame matching raw schema
        input_data = {
            'likes': likes,
            'comments': comments,
            'subscriber_count': subscribers,
            'title_length': title_length,
            'tags_count': tags_count,
            'duration_sec': duration_sec,
            'upload_hour': upload_hour,
            'is_weekend': is_weekend,
            f'cat_{category}': 1
        }
        input_df = pd.DataFrame([input_data])

        # Predict probability & apply optimal threshold (0.7205)
        prob = pipeline.predict_proba(input_df)[:, 1][0]
        is_viral = prob >= optimal_threshold

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Virality Probability", f"{prob:.2%}")
        res_col2.metric("Decision Cutoff", f"{optimal_threshold:.2%}")

        if is_viral:
            res_col3.success("PREDICTION: VIRAL HIT (Class 1)")
        else:
            res_col3.warning("PREDICTION: STANDARD / NON-VIRAL (Class 0)")

# --- TAB 2: MODEL EXPERIMENT REGISTRY ---
with tab2:
    st.subheader("Sprint Model Progression & Benchmark History")
    st.markdown(f"**Active Production Champion:** Tuned XGBoost (`ROC-AUC: {metrics['roc_auc']:.4f}` | `Optimal F1: {metrics['best_f1']:.4f}`)")

    history_df = pd.DataFrame([
        {"Day": "Day 6", "Model": "Dummy Classifier", "Features / Setup": "Uniform Random Baseline", "ROC-AUC": 0.4940, "Status": "Retired"},
        {"Day": "Day 6", "Model": "Balanced Logistic Regression", "Features / Setup": "Scaled Ratios (Leakage-Free)", "ROC-AUC": 0.5221, "Status": "Retired"},
        {"Day": "Day 7", "Model": "Decision Tree Classifier", "Features / Setup": "max_depth=5, balanced weights", "ROC-AUC": 0.5005, "Status": "Retired"},
        {"Day": "Day 7", "Model": "Random Forest Classifier", "Features / Setup": "100 Trees, max_depth=10", "ROC-AUC": 0.6728, "Status": "Retired"},
        {"Day": "Day 8", "Model": "LightGBM Baseline", "Features / Setup": "Log-Transformed Engagement Ratios", "ROC-AUC": 0.9865, "Status": "Runner-Up"},
        {"Day": "Day 8–10", "Model": "Tuned XGBoost Champion", "Features / Setup": "Log-Ratios + 0.7205 Optimal Threshold", "ROC-AUC": 0.9862, "Status": "PRODUCTION"}
    ])

    st.dataframe(history_df, use_container_width=True, hide_index=True)
    st.bar_chart(history_df.set_index("Model")["ROC-AUC"])