"""Importing libraries"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

"""Storing data with pandas"""
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

"""Storing data with numpy"""
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""Setting up best score"""
best_score = 0
for epochs in range(30):

    """Preparing testing and training"""
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    """Getting accuracy"""
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    """Saves model if it was the best score out of 30 epochs"""
    if accuracy > best_score:
        best_score = accuracy

    """"Writing a pickle file"""
    with open("studentmodel.pickle", "wb") as file:
        pickle.dump(linear, file)

"""Open pickle"""
pickle_open = open("studentmodel.pickle", "rb")

"""Load pickle"""
linear = pickle.load(pickle_open)

"""A view to see where the algorithm fails"""
print("Coefficient: \n", linear.coef_)
print("Intercept \n", linear.intercept_)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
