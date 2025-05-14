import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ================== Data Preprocessing Function ==================


def preprocess_data(df, label_encoders=None, num_imputer=None, training=True):
    # Replace values in categorical columns
    if "PreferredLoginDevice" in df.columns:
        df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(
            "Phone", "Mobile Phone"
        )
    if "PreferredPaymentMode" in df.columns:
        df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
            "CC", "Credit Card"
        )

    # Drop ID column
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop(
        "Churn", errors="ignore"
    )
    cat_cols = df.select_dtypes(include="object").columns

    # Impute missing values
    if training:
        num_imputer = SimpleImputer(strategy="mean")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    else:
        df[num_cols] = num_imputer.transform(df[num_cols])

    # Encode categorical columns
    if training:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in cat_cols:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    return df, label_encoders, num_imputer


# ================== Load & Preprocess Data ==================

df_raw = pd.read_excel(
    r"D:\PYTHON 3\ICTAK Python3\customer_churn_prediction\data\raw\E Commerce Dataset.xlsx",
    sheet_name="E Comm",
)

df_cleaned, label_encoders, num_imputer = preprocess_data(df_raw, training=True)

output_dir = "D:/PYTHON 3/ICTAK Python3/customer_churn_prediction/data/cleaned"
os.makedirs(output_dir, exist_ok=True)

df_cleaned.to_excel(os.path.join(output_dir, "E_Commerce_Cleaned.xlsx"), index=False)
print("✅ Cleaned dataset saved successfully.")


X = df_cleaned.drop(columns=["Churn"])
y = df_cleaned["Churn"]

# ================== Handle Class Imbalance ==================

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ================== Train-Test Split ==================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ================== Random Forest Tuning & Training ==================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_grid, cv=5, scoring="roc_auc", n_iter=10, verbose=2, n_jobs=-1
)
random_search.fit(X_train, y_train)
print("✅ Best Hyperparameters:", random_search.best_params_)

# ================== Feature Importance ==================

best_rf = RandomForestClassifier(**random_search.best_params_, random_state=42)
best_rf.fit(X_train, y_train)

importances = best_rf.feature_importances_
feature_names = best_rf.feature_names_in_
sorted_indices = np.argsort(importances)[::-1]
top_n = 10
top_features = feature_names[sorted_indices[:top_n]]

print("\n✅ Top Feature Importances:")
for i in range(top_n):
    print(f"{feature_names[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")

X_train_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]

# ================== Final Random Forest Model ==================

final_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
final_model.fit(X_train_reduced, y_train)

y_pred = final_model.predict(X_test_reduced)
print(
    "\n🔸 [Random Forest] Classification Report:\n",
    classification_report(y_test, y_pred),
)
print("🔸 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(
    "🔸 ROC-AUC Score:",
    roc_auc_score(y_test, final_model.predict_proba(X_test_reduced)[:, 1]),
)

# ================== Other Models (Evaluation Only) ==================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_reduced, y_train)
y_pred_lr = lr.predict(X_test_reduced)
print(
    "\n🔸 [Logistic Regression] Classification Report:\n",
    classification_report(y_test, y_pred_lr),
)
print(
    "🔸 ROC-AUC Score:", roc_auc_score(y_test, lr.predict_proba(X_test_reduced)[:, 1])
)

# SVM
svm = SVC(probability=True)
svm.fit(X_train_reduced, y_train)
y_pred_svm = svm.predict(X_test_reduced)
print("\n🔸 [SVM] Classification Report:\n", classification_report(y_test, y_pred_svm))
print(
    "🔸 ROC-AUC Score:", roc_auc_score(y_test, svm.predict_proba(X_test_reduced)[:, 1])
)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train_reduced, y_train)
y_pred_xgb = xgb.predict(X_test_reduced)
print(
    "\n🔸 [XGBoost] Classification Report:\n", classification_report(y_test, y_pred_xgb)
)
print(
    "🔸 ROC-AUC Score:", roc_auc_score(y_test, xgb.predict_proba(X_test_reduced)[:, 1])
)

# ================== Save Final Random Forest Model ==================

model_bundle = {
    "model": final_model,
    "features": list(top_features),
    "label_encoders": label_encoders,
    "num_imputer": num_imputer,
}

joblib.dump(model_bundle, "churn_model_bundle.pkl")
print("\n✅ Final Random Forest model saved as 'churn_model_bundle.pkl'")
