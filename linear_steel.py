import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("steel_strength.csv")

# 2. Select features and target
target_col = "yield strength"   #  choose correct column
X = df.drop(columns=["formula", "yield strength", "tensile strength", "elongation"])
y = df[target_col]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# 5. Predict
y_pred = linreg.predict(X_test)

# 6. Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 7. Plot actual vs predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Yield Strength")
plt.ylabel("Predicted Yield Strength")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--")
plt.show()

# 8. Residuals
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual (Actual – Predicted)")
plt.title("Residual distribution: Linear Regression")
plt.show()

# 9. Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linreg.coef_
})
print(coef_df.sort_values(by="Coefficient", ascending=False))
