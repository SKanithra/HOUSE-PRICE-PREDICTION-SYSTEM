
# ADVANCED HOUSE PRICE PREDICTION SYSTEM


#  Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load Dataset
data = pd.read_csv("2_dataset.csv")

print("\n========== ADVANCED HOUSE PRICE PREDICTION ==========")
print("\n📊 Dataset Preview:\n", data.head())

#  Check Missing Values
print("\n🔍 Missing Values:\n", data.isnull().sum())

#  Features and Target
X = data[['area', 'bedrooms', 'location']]
y = data['price']

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#  Preprocessing (OneHotEncoding)
categorical = ['location']
numeric = ['area', 'bedrooms']

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical),
    ("num", "passthrough", numeric)
])

#  Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# Train + Evaluate Models
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross Validation
    cv_score = cross_val_score(pipeline, X, y, cv=5, scoring='r2').mean()

    results[name] = {
        "model": pipeline,
        "MAE": mae,
        "R2": r2,
        "CV_R2": cv_score
    }

# Display Results
print("\n📈 Model Comparison:")
print("----------------------------------------")
for name, res in results.items():
    print(f"\n{name}")
    print(f"MAE: {res['MAE']:,.2f}")
    print(f"R2 Score: {res['R2']:.2f}")
    print(f"Cross-Validation R2: {res['CV_R2']:.2f}")

#  Choose Best Model
best_model_name = max(results, key=lambda x: results[x]['CV_R2'])
best_model = results[best_model_name]["model"]

print(f"\n🏆 Best Model Selected: {best_model_name}")

#  Visualization
y_pred_best = best_model.predict(X_test)

plt.scatter(y_test, y_pred_best)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"{best_model_name} - Actual vs Predicted")
plt.show()

#  Prediction
new_house = pd.DataFrame([[1400, 3, 'A']],
                         columns=['area', 'bedrooms', 'location'])

predicted_price = best_model.predict(new_house)

print("\n🏠 Final Prediction:")
print("----------------------------------------")
print("Area = 1400 sq.ft | Bedrooms = 3 | Location = A")
print(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
