#import neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("air_quality_data.csv")  # Ensure you have an air quality dataset

# Display first few rows
print(df.head())
# Check for missing values
df.fillna(df.mean(), inplace=True)

# Selecting relevant features
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'Temperature', 'Humidity']
target = 'AQI'
# Splitting dataset into training and testing
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predicting on test data
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
# Plot actual vs predicted values
plt.figure(figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, s=100, color='blue', edgecolor='black')
plt.plot(y_test, y_test, color='red', lw=2)
plt.xlabel("Actual AQI", fontsize=12)
plt.ylabel("Predicted AQI", fontsize=12)
plt.title("Actual vs Predicted AQI", fontsize=15)
plt.show()
