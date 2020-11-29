import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

"""Importing data, change dir"""
data = pd.read_csv("C:/Users/SMXaS/PycharmProjects/TensorMachine/K_Nearest_Neighbors/car.data")

"""Takes the labels and codes into values"""
process = preprocessing.LabelEncoder()
buying = process.fit_transform(list(data["buying"]))
maint = process.fit_transform(list(data["maint"]))
door = process.fit_transform(list(data["door"]))
persons = process.fit_transform(list(data["persons"]))
lug_boot = process.fit_transform(list(data["lug_boot"]))
safety = process.fit_transform(list(data["safety"]))
class_data = process.fit_transform(list(data["class"]))

"""What data predicting"""
predict = "class"

"""Setting up X and Y, storing data"""
X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(class_data)

"""Splitting data into X and Y train, test"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""Amount of neighbors for accuracy"""
model = KNeighborsClassifier(n_neighbors=5)

"""Training the model"""
model.fit(x_train, y_train)

"""Setting Up accuracy"""
accuracy = model.score(x_test, y_test)

"""Printing out accuracy"""
print(accuracy)

"""Matrix for  model prediction """
model_predicted = model.predict(x_test)
values = ["inaccurate", "accurate", "not bad", "amazing"]

"""Printing out the matrix model forecast"""
for forecast in range(len(model_predicted)):
    print("The algorithm has predicted: ", values[model_predicted[forecast]],
          "Data : ", x_test[forecast],
          "Actual value, output of the algorithm: ", values[y_test[forecast]])
"""" neighbor = model.kneighbors([x_test[predicting]], 9)
    print("N: ", neighbor)"""
