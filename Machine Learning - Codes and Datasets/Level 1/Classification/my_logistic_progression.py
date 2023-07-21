import pandas as pd
dataset = pd.read_csv('data.csv')
# print(dataset.head(10))

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
# print(x_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)
# print(model)

y_pred = model.predict(sc.transform(x_test))
# print(y_pred)

# print(model.predict(sc.transform([[1,2,3,4,5,6,7,8,9]])))

from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, y_pred))

#print((84+47)/(84+47+3+3))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

