import matplotlib.pyplot as plt
import numpy

from sklearn import linear_model

# X = numpy.arange(-100, 100, 0.1)
# Y = X**3
X = [1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 11, 14, 17, 19, 20, 21]
Y = [1, 1, 3, 4, 5, 10, 2, 4, 6, 7, 13, 14, 13, 16, 14, 19]
max_X = max(X)
min_X = min(X)
train_data_X = map(lambda x: [x], list(X[:-5]))
train_data_Y = list(Y[:-5])
test_data_X = map(lambda x: [x], list(X[-5:]))
test_data_Y = list(Y[-5:])

reg = linear_model.LinearRegression()
reg.fit(train_data_X, train_data_Y)
m = reg.coef_[0]
b = reg.intercept_
# reg.fit(X, Y)
plt.scatter(X, Y, color='blue')
plt.plot([min_X, max_X], [b, m*max_X + b], 'r')
plt.title('Linear regression', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()

# values = [(3, 10), (4, 10), (5, 10), (6, 10), (7, 11), (8, 12)]