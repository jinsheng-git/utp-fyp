# Default XGBoost configuration #
# -------------------------------------------------- #
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data
data = pd.read_excel('Logistics_Data.xlsx')
df = data

# Extract Year, Month, Day, Hour, DayOfWeek from col Timestamp
# Then drop col Timestamp
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df = df.drop(columns=['Timestamp'])

# Drop potential data leak col
df = df.drop(columns=['Logistics_Delay_Reason'], errors='ignore')
df = df.drop(columns=[col for col in df.columns if 'Logistics_Delay_Reason' in col or 'Traffic_Status' in col], errors='ignore')

# One-Hot Encoding on categorical col
df = pd.get_dummies(df, columns=['Asset_ID', 'Shipment_Status'], drop_first=True)

# Define Features (x) and Target (y)
X = df.drop('Logistics_Delay', axis=1)
y = df['Logistics_Delay']

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost model
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=1,
    random_state=42
)

xgb.fit(X_train, y_train)

# Predict and Evaluate
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nXGBoost Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# -------------------------------------------------- #
# Tuned XGBoost #
# -------------------------------------------------- #
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Data
data = pd.read_excel('Logistics_Data.xlsx')
df = data

# Extract Year, Month, Day, Hour, DayOfWeek from col Timestamp
# Then drop col Timestamp
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df = df.drop(columns=['Timestamp'])

# Drop potential data leak col
df = df.drop(columns=['Logistics_Delay_Reason'], errors='ignore')
df = df.drop(columns=[col for col in df.columns if 'Logistics_Delay_Reason' in col or 'Traffic_Status' in col], errors='ignore')

# One-Hot Encoding on categorical col
df = pd.get_dummies(df, columns=['Asset_ID', 'Shipment_Status'], drop_first=True)

# Define Features (x) and Target (y)
X = df.drop('Logistics_Delay', axis=1)
y = df['Logistics_Delay']

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid search setup
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# XGBoost model evaluation
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTuned XGBoost Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# -------------------------------------------------- #
