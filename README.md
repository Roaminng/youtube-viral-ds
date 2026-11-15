# YouTube Viral Video Predictor & Model Experiment Tracker

An end-to-end Machine Learning classification pipeline and interactive Streamlit web application predicting whether a YouTube video will reach viral status (top 20% of views) across a 12,000-video dataset.

## Live Demo
- **Interactive App:** [Add your Streamlit Community Cloud URL after Step 3]

## Project Highlights & Methodology
1. **Data Hygiene & Leakage Auditing:** Audited features and removed `log_views` to prevent target leakage, ensuring realistic inference using engagement velocity and channel metadata.
2. **Feature Engineering:** Engineered logarithmic engagement transformations (`log_likes`, `log_comments`, `log_subscribers`) and relative channel ratios (`log_like_per_sub`, `log_engagement_sum`).
3. **Model Progression & Benchmarking:**
   - Dummy Baseline: `0.4940 ROC-AUC`
   - Balanced Logistic Regression: `0.5221 ROC-AUC`
   - Random Forest Classifier: `0.6728 ROC-AUC`
   - **Tuned XGBoost Champion:** **`0.9862 ROC-AUC`**
4. **Explainability & Threshold Optimization:** Validated key drivers via **SHAP** (`log_likes` and `log_engagement_sum`) and tuned the decision cutoff to **`0.7205`**, achieving **`0.8646 F1-score`** and **`95.24% precision`** on viral predictions.
5. **Production Deployment:** Serialized the custom `ViralFeatureEngineer` transformer and XGBoost classifier into a unified Scikit-Learn `Pipeline` (`youtube_viral_champion_v1.joblib`) served via **Streamlit** (`app.py`).

## Tech Stack
- **Data & ML:** Python, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, SHAP, Joblib
- **App & Deployment:** Streamlit, Streamlit Community Cloud, Git/GitHub