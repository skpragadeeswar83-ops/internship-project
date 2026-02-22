import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Create dummy data if city_day.csv is missing
if not os.path.exists('city_day.csv'):
    data = {
        'PM2.5': np.random.uniform(10, 300, 500),
        'PM10': np.random.uniform(20, 400, 500),
        'NO2': np.random.uniform(5, 100, 500),
        'CO': np.random.uniform(0.1, 5, 500),
        'SO2': np.random.uniform(2, 50, 500),
        'O3': np.random.uniform(5, 100, 500),
        'AQI': np.random.uniform(20, 500, 500)
    }
    pd.DataFrame(data).to_csv('city_day.csv', index=False)
    print("Dataset created successfully.")

# 1. Data Collection & Preprocessing
df = pd.read_csv('city_day.csv')
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
X = df[features]
y = df['AQI']

# 2. Training Process
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# 3. Prediction & Evaluation
predictions = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}")

# 4. Visualization for Slide 13
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('AQI Prediction: Actual vs Predicted')
plt.show()