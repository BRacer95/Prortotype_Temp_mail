import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({'Size': [150, 200, 250, 300, 350], 'Price': [1200, 1250, 1300, 1350, 1400]})
X, y = data[['Size']], data['Price']

model = LinearRegression().fit(X, y)
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("House Price Prediction")
plt.show()