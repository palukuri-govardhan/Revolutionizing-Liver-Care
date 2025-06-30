import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# --- Load dataset ---
try:
    df = pd.read_csv('HealthCareData.csv')
except FileNotFoundError:
    print("Error: HealthCareData.csv not found.")
    exit()

# --- Detect target column ---
target_column = None
for col in df.columns:
    if 'suffering from liver' in col.lower() and 'cirrosis' in col.lower():
        target_column = col
        break

if target_column:
    df = df.rename(columns={target_column: 'LiverCirrhosis'})
    df['LiverCirrhosis'] = df['LiverCirrhosis'].map({'YES': 1, 'NO': 0, 'yes': 1, 'no': 0})
    df['LiverCirrhosis'] = df['LiverCirrhosis'].fillna(0).astype(int)
else:
    print("Target column not found.")
    exit()

# --- Preprocess binary columns ---
binary_cols = ['Obesity', 'Family history of cirrhosis/ hereditary', 'USG Abdomen (diffuse liver or  not)',
               'Gender', 'Diabetes Result', 'Hepatitis B infection', 'Hepatitis C infection']

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'YES': 1, 'NO': 0, 'yes': 1, 'no': 0,
                               'positive': 1, 'negative': 0,
                               'Positive': 1, 'Negative': 0,
                               'male': 1, 'female': 0})

# --- Handle Blood Pressure ---
if 'Blood pressure (mmhg)' in df.columns:
    df['Blood pressure (mmhg)'] = df['Blood pressure (mmhg)'].astype(str)
    bp_split = df['Blood pressure (mmhg)'].str.split('/', expand=True)
    df['SystolicBP'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['DiastolicBP'] = pd.to_numeric(bp_split[1], errors='coerce')
    df.drop(columns=['Blood pressure (mmhg)'], inplace=True)

# --- Convert all object columns to numeric (if possible) ---
for col in df.columns:
    if col != 'LiverCirrhosis' and df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Fill missing numeric data with mean ---
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

# --- Drop S.NO and other object columns ---
df = df.drop(columns=['S.NO'], errors='ignore')
df = df.select_dtypes(include=[np.number])

# --- Define features and target ---
X = df.drop(columns=['LiverCirrhosis'])
y = df['LiverCirrhosis']

# --- Standardize and clean ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled = X_scaled.dropna(axis=1, how='all').fillna(X_scaled.mean()).dropna()
y = y.loc[X_scaled.index]

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Model evaluation function ---
def models_eval_mm(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Logistic Regression CV': LogisticRegressionCV(cv=5, max_iter=1000, random_state=42),
        'XGBoost Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Ridge Classifier': RidgeClassifier(),
        'KNN Classifier': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }

    trained_models = {}
    print("\n🎯 Performance Testing (Before Tuning):")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Train Score: {train_score:.4f}")
        print(f"Test Score: {test_score:.4f}")
        trained_models[name] = model

    return trained_models

# --- Run performance test ---
trained_models = models_eval_mm(X_train, X_test, y_train, y_test)

# --- Hyperparameter tuning ---
print("\n🔧 Hyperparameter Tuning with GridSearchCV...\n")

# Logistic Regression
lr_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5)
lr_grid.fit(X_train, y_train)
print(f"Best Logistic Regression Params: {lr_grid.best_params_} | CV Score: {lr_grid.best_score_:.4f}")

# KNN
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_train, y_train)
print(f"Best KNN Params: {knn_grid.best_params_} | CV Score: {knn_grid.best_score_:.4f}")

# Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
print(f"Best Random Forest Params: {rf_grid.best_params_} | CV Score: {rf_grid.best_score_:.4f}")

# XGBoost
xgb_params = {'max_depth': [3, 5], 'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        xgb_params, cv=5)
xgb_grid.fit(X_train, y_train)
print(f"Best XGBoost Params: {xgb_grid.best_params_} | CV Score: {xgb_grid.best_score_:.4f}")

# --- Evaluate tuned models ---
print("\n📊 Evaluating Tuned Models on Test Set:")

lr_test = lr_grid.best_estimator_.score(X_test, y_test)
knn_test = knn_grid.best_estimator_.score(X_test, y_test)
rf_test = rf_grid.best_estimator_.score(X_test, y_test)
xgb_test = xgb_grid.best_estimator_.score(X_test, y_test)

print(f"Tuned Logistic Regression Test Score: {lr_test:.4f}")
print(f"Tuned KNN Test Score: {knn_test:.4f}")
print(f"Tuned Random Forest Test Score: {rf_test:.4f}")
print(f"Tuned XGBoost Test Score: {xgb_test:.4f}")
