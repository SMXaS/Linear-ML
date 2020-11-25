import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

"""Importing data, change dir"""
data = pd.read_csv("C:/Users/SMXaS/PycharmProjects/TensorMachine/K_Nearest_Neighbors/car.data")
print(data.head())

"""Takes the Labels and codes into values"""
process = preprocessing.LabelEncoder()
buying = process.fit_transform(list(data["buying"]))
maint = process.fit_transform(list(data["maint"]))
door = process.fit_transform(list(data["door"]))
persons = process.fit_transform(list(data["persons"]))
lug_boot = process.fit_transform(list(data["lug_boot"]))
safety = process.fit_transform(list(data["safety"]))
class_data = process.fit_transform(list(data["class"]))

predict = "class"
print(buying)