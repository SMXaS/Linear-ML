"""Importing libraries"""
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

"""Getting data"""
cancer = datasets.load_breast_cancer()

"""Printing data"""
#print(cancer.feature_names)
#print(cancer.target_names)

"""Assigning data to x and y"""
x = cancer.data
y = cancer.target

"""Best score of accuracy"""
best_score = 0

"""Looping the accuracy 30 times"""
for score in range(30):

    """Training and testing, change test size"""
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    """Printing out training data"""
    #print(x_train, y_train)

    """Data"""
    classes_data = ["malignant", "benign"]

    """Classifier with parameter, you can change parameter"""
    classifier = svm.SVC(kernel="linear")
    classifier.fit(x_train, y_train)

    """Running the test"""
    prediction_y = classifier.predict(x_test)

    """Getting the accuracy score"""
    accuracy = metrics.accuracy_score(y_test, prediction_y)

    """Printing out the accuracy"""
    print(accuracy)

    """If accuracy is the new best score = save"""
    if accuracy > best_score:
        best_score = accuracy
