from statistics import LinearRegression
import pandas as pd
from sklearn.cluster import k_means
dataset = pd.read_excel('data.xlsx')

# print(dataset.head())

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# data preprocessing, setting up data for training and testing
# add data to pool and organize for AI brain to pull from

from sklearn.model_selection  import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

# print("x train: ", x_train)
# print("x_test: ", x_test)
# print("y_train: ", y_train)
# print("y_test: ", y_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# train the model

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# print(y_pred)


# make prediction of a single data point
single = model.predict([[15,40,1000,75]])
# print(single)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

k = x_test.shape[1]
n = x_test.shape[0]
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
print(adj_r2)