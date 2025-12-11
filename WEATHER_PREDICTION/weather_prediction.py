import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras import models, layers

df = pd.read_csv("/content/drive/MyDrive/ML/seattle-weather.csv")
print(df.head())

df["temperature"] = (df["temp_max"] + df["temp_min"]) / 2

X = df.drop(["temperature", "date", "weather"], axis=1)
y = df["temperature"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

epochs = 50
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

min_val_loss = min(history.history["val_loss"])
best_epoch = history.history["val_loss"].index(min_val_loss) + 1

print("\nBest Epoch (Lowest Loss):", best_epoch)
print("Minimum Validation Loss :", min_val_loss)

y_pred = model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

print("\n----- FINAL PERFORMANCE METRICS -----")
print("MAE :", MAE)
print("MSE :", MSE)
print("RMSE:", RMSE)
print("R² Score:", R2)
print("Prediction Accuracy (%):", R2 * 100)

plt.figure(figsize=(8, 6))

plt.scatter(y_test.values[:100], y_pred[:100], color='blue', label='Predicted vs Actual', alpha=0.6)

plt.plot([min(y_test.values[:100]), max(y_test.values[:100])],
         [min(y_test.values[:100]), max(y_test.values[:100])],
         color='red', linestyle='--', label='Perfect Prediction')

plt.title("Actual vs Predicted Temperature ")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.legend()
plt.grid(True)
plt.show()

sample = df.drop(["temperature", "date", "weather"], axis=1).iloc[-1].values.reshape(1, -1)

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nPredicted Temperature =", prediction[0][0])