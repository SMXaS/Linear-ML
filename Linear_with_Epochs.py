"""Importing Libraries"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

"""Storing Data with Pandas"""
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

"""Storing Data with Numpy"""
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""Setting up Best Score"""
best_score = 0
for epochs in range(30):

    """Preparing Testing and Training"""
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    """Getting Accuracy"""
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    """Saves model if it was the best score out of 30 epochs"""
    if accuracy > best_score:
        best_score = accuracy

    """"Writing a Pickle file"""
    with open("studentmodel.pickle", "wb") as file:
        pickle.dump(linear, file)

"""Open Pickle"""
pickle_open = open("studentmodel.pickle", "rb")

"""Load Pickle"""
linear = pickle.load(pickle_open)

"""A view to see where the Algorithm fails"""
print("Coefficient: \n", linear.coef_)
print("Intercept \n", linear.intercept_)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
