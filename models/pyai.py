import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([40, 50, 60, 70, 80])

model = LinearRegression() # Empty brain creation
model.fit(x, y) # learn the relationship between x and y
# score = 10 × hours + 30 is what model learnt

# prediction = model.predict([[6]]) # Model expects 2D input, so we give 2D input
# print(prediction[0])

y_predict = model.predict(x)
print(y_predict)

plt.scatter(x, y)
plt.plot(x, y_predict)
plt.xlabel("Hours studied")
plt.ylabel("Marks expected")
plt.title("Marks vs Study hours")
plt.show()
