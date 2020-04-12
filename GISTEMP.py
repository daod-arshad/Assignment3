# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('GISTEMP.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Mean Temperature vs Year GISTEMP (Decision Tree Regression)')
plt.xlabel('Year')
plt.ylabel('Mean temperature')
plt.show()


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =10)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Mean Temperature vs Year GISTEMP (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Mean temperature')
plt.show()

# Visualising the Training Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Mean Temperature vs Year GISTEMP Traing set (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Mean temperature')
plt.show()
# Visualising the Test Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Mean Temperature vs Year GISTEMP Test Set (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Mean temperature')
plt.show()

print('2016 and 2017 temperatures respectively')
print('TEMPERATURE ACCORDING TO Polynomial Regression')
print(lin_reg_2.predict(poly_reg.fit_transform([[2016],[2017]])))
print('TEMPERATURE ACCORDING TO GISTEMP Decision TREE')
y_pred = regressor.predict([[2016],[2017]])
print(y_pred)
