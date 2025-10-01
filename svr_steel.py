import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Load
CSV_PATH = "steel_strength.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH} in {os.getcwd()}")

df = pd.read_csv(CSV_PATH)

# 2) Pick target column (auto-detect any column containing 'strength')
#    This also trims stray spaces in headers.
df.columns = df.columns.str.strip()
strength_like = [c for c in df.columns if "strength" in c.lower()]

if not strength_like:
    raise ValueError(
        "No column containing 'strength' found. "
        f"Available columns: {list(df.columns)}\n"
        "Tip: rename the actual target column to 'strength' or update the code below."
    )

# If there are multiple 'strength' columns, use the first—adjust if needed.
target_col = strength_like[0]
print(f"Using target column: {target_col}")

X = df.drop(columns=[target_col])
y = df[target_col]

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Preprocess
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# 5) Model pipeline
pipe = Pipeline(steps=[
    ("pre", preprocessor),
    ("svr", SVR(kernel="rbf")),
])

# 6) Hyperparameter search
param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.01, 0.1, 0.5, 1.0],
    "svr__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
best_model = grid.best_estimator_

# 7) Predict & evaluate
y_pred = best_model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\nSVR Performance")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# 8) Plots (matplotlib only)
# Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
vmin = min(np.min(y_test), np.min(y_pred))
vmax = max(np.max(y_test), np.max(y_pred))
plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVR: Actual vs Predicted")
plt.tight_layout()
plt.savefig("svr_actual_vs_pred.png", dpi=150)
plt.show()

# Residuals
residuals = y_test.values - y_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (Actual – Predicted)")
plt.title("Residual Distribution: SVR")
plt.tight_layout()
plt.savefig("svr_residuals.png", dpi=150)
plt.show()

# 9) (Optional) Save predictions
out = pd.DataFrame({
    "y_test": np.array(y_test),
    "y_pred": y_pred
})
out.to_csv("svr_predictions.csv", index=False)
print("\nSaved: svr_actual_vs_pred.png, svr_residuals.png, svr_predictions.csv")
