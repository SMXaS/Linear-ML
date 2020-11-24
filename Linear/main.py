"""Importing Libraries"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

"""Storing Data with Pandas"""
data = pd.read_csv("C:/Users/SMXaS/PycharmProjects/TensorMachine/Linear/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

"""Storing Data with Numpy"""
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

"""Saving X and Y, because it is used in 53 - 57 lines"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

''' Commented out when the best accuracy was found (95%), 
    if you want to keep going, uncomment and run the training.

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
        pickle.dump(linear, file) '''

"""Open Pickle"""
pickle_open = open("C:/Users/SMXaS/PycharmProjects/TensorMachine/Linear/studentmodel.pickle", "rb")

"""Load Pickle"""
linear = pickle.load(pickle_open)

"""A view to see where the Algorithm fails"""
print("Coefficient: \n", linear.coef_)
print("Intercept \n", linear.intercept_)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

"""Making a Grid View"""
"""Picked number can be Changed to any other data of line 12"""
picked_number = "G1"
style.use("ggplot")
pyplot.scatter(data[picked_number], data["G3"])
pyplot.xlabel(picked_number)
pyplot.ylabel("Final Grade")
pyplot.show()
