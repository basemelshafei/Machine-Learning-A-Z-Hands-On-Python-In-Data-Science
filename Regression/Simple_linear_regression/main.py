import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

print(data)
print(X)
print(Y)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
print(train_x)
print(train_y)
print(test_x)
print(test_y)

regressor = LinearRegression()
regressor.fit(train_x, train_y)
regressor.fit(train_x, train_y)
pred_y = regressor.predict(test_x)


plt.scatter(train_x, train_y, color='red', alpha=0.4)
plt.plot(train_x, regressor.predict(train_x), color='blue')
plt.title(' Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(test_x, test_y, color='red', alpha=0.4)
plt.plot(test_x, pred_y, color='blue')
plt.title(' Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


plt.show()










