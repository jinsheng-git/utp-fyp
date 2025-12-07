# Machine Learning in Supply Chain Risk Prediction
My Final Year Project at University Technology PETRONAS (January 2025 - August 2025)

# Description
This is a machine learning model building focuses on Random Forest (RF), XGBoost (XGB), and Long Short Term Memory (LSTM).

Three model *rf_lstm*, *random_forest*, and *xgboost* in separate .py files.

### *rf_lstm.py*
An ensemble machine model that...
1. Initial idea to utilise RF feature selection (x) for LSTM target prediction (y).
2. RF phase succeed, able to determine the top n important features.
3. LSTM phase failed, dataset chosen lack of efficient time-series pattern. Result remained the same after data processing on time-related data.
4. Model building attempt stopped here, then moved on to standalone tree-based model building to achieve project's goal.
<br><br>

### *random_forest.py*
Included default RF configuration and tuned RF configuration. Nothing special to mention, can try different tuning.
<br><br>

### *xgboost.py*
Included default XGB configuration and tuned XGB configuration. Nothing special to mention, can try different tuning.
<br><br>

# Installation
Any Python notebook will work, recommend to use Google Collab, I separated each block by # ---------- #.
<br><br>
**Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow

**Dataset:** https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset
