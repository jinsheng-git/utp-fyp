# Final Year Project at Universiti Teknologi PETRONAS

**Title:** Machine Learning in Supply Chain Risk Prediction <br><br>

**Summary**
- I came up with the idea of utilising the ability of Random Forest to select important features (x) that contribute to Long Short Term Memory target prediction (y).
- For rf_lstm.py, the rf feature selection phase succeed but the lstm prediction phase failed. Reason being is that the dataset chosen does not have efficient time-series pattern that lstm needed. Hence, the project changes direction to the comparison of Random Forest and XGBoost.
- For random-forest.py, it contains the default configuration and tuned configuration.
- For xgboost.py, it contains the default configuration and tuned configuration. <br><br>

**Requirements**
- **Dataset:** https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset
- **Tools:** Google Colab
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow <br><br>

**Reminders**
- This project conducted with Google Colab to avoid any operating system and python libraries constraint.
- Inside the rf_lstm.py, random-forest.py, and xgboost.py there is section separated by "# ---------- #".
