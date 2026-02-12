import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# Features: [hours studied, sleep hours]
# Adds an extra layer of security and now the ideality is defined by a 3D plane which compares the noise of data
x = np.array([
    [1, 5], [2, 6], [3, 6], [4, 7], [5, 8]
])
y = np.array([50, 60, 70, 80, 90])

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("Input:", x_test.flatten())
print("Actual Value:", y_test)
print("Predictions:", predictions[0])
print(model.coef_)
print(model.intercept_)

# print("Accuracy:", r2_score(y_test, predictions))
# R2 score is the accuracy index of the model, 1.0 -> perfect, 0 -> useless, -ve -> worse than guessing
