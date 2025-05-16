import os
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

#    Data Preprocessing Function
# ================================
def preprocess_data(df, label_encoders=None, num_imputer=None, training=True):
    """
    Cleans, transforms, and encodes the input dataframe for model training or prediction.
    """

    # ---- Replace values for consistency in categorical columns ----
    if "PreferredLoginDevice" in df.columns:
        df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace("Phone", "Mobile Phone")
    if "PreferredPaymentMode" in df.columns:
        df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace("CC", "Credit Card")
        df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace("COD", "Cash on Delivery")
    if "PreferedOrderCat" in df.columns:
        df["PreferedOrderCat"] = df["PreferedOrderCat"].replace("Mobile", "Mobile Phone")

    # ---- Drop columns not needed for model training ----
    drop_cols = ["Gender", "MaritalStatus", "PreferredLoginDevice", "PreferredPaymentMode"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # ---- Drop identifier column ----
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # ---- Separate numerical and categorical columns ----
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("Churn", errors="ignore")
    cat_cols = df.select_dtypes(include="object").columns

    # ---- Impute missing numerical values ----
    if training:
        num_imputer = SimpleImputer(strategy="mean")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    else:
        df[num_cols] = num_imputer.transform(df[num_cols])

    # ---- Save cleaned data before encoding (for analysis) ----
    if training:
        unencoded_path = "data/cleaned/E_Commerce_Cleaned_Before_Encoding.xlsx"
        os.makedirs(os.path.dirname(unencoded_path), exist_ok=True)
        df.to_excel(unencoded_path, index=False)
        print("✅ Cleaned dataset (before encoding) saved successfully.")

    # ---- Encode categorical features using Label Encoding ----
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


#         Load Raw Dataset
# ================================
df_raw = pd.read_excel("data/raw/E Commerce Dataset.xlsx", sheet_name="E Comm")

#     Clean & Encode Dataset
# =================================
df_cleaned, label_encoders, num_imputer = preprocess_data(df_raw, training=True)

X = df_cleaned.drop(columns=["Churn"])
y = df_cleaned["Churn"]


#      Handle Imbalanced Data
# ================================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


#       Train/Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

#     Random Forest Hyperparams
# ================================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 12, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 6],
    "class_weight": ["balanced"],
}


#    Hyperparameter Tuning (RSCV)
# ================================
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_grid, cv=5, scoring="roc_auc", n_iter=10, verbose=2, n_jobs=-1
)
random_search.fit(X_train, y_train)


#     Train Best Random Forest
# ================================
best_rf = RandomForestClassifier(
    **random_search.best_params_,
    max_samples=0.8,
    random_state=42
)
best_rf.fit(X_train, y_train)


#        Model Evaluation
# ================================
required_features = list(best_rf.feature_names_in_)
X_train_reduced = X_train[required_features]
X_test_reduced = X_test[required_features]

train_score = best_rf.score(X_train_reduced, y_train)
test_score = best_rf.score(X_test_reduced, y_test)
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# ---- Cross-validation ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_rf,
    X_resampled[required_features],
    y_resampled,
    cv=cv,
    scoring="accuracy"
)
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


#        Save Model Bundle
# ================================
model_bundle = {
    "model": best_rf,
    "features": required_features,
    "label_encoders": label_encoders,
    "num_imputer": num_imputer,
}

joblib.dump(model_bundle, "churn_model_bundle.pkl")
print("✅ Model saved as 'churn_model_bundle.pkl'")
