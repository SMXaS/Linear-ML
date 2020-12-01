"""Importing libraries"""
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

"""Loading data"""
digits = load_digits()

"""Scaling data"""
data = scale(digits.data)

"""Labeling"""
y = digits.target

"""Setting amount of numbers, can be changed to int"""
k_means_length = len(np.unique(y))

"""Shaping the data"""
samples, features = data.shape

"""Function for training the classifiers, 
filled with parameters 
and gives different scores"""


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


"""Setting up classifiers"""
classifier = KMeans(n_clusters=k_means_length, init="random", n_init=10)
bench_k_means(classifier, "1", data)
