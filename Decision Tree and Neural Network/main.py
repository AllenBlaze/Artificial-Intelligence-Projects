import numpy as np
#import matplotlib.pyplot as plt
import sys
import load_datasets
# import NeuralNet  # importer la classe du Réseau de Neurones
# import DecisionTree  # importer la classe de l'Arbre de Décision
# importer d'autres fichiers et classes si vous en avez développés
# importer d'autres bibliothèques au besoin, sauf celles qui font du machine learning

from DecisionTree import DecisionTree

iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(0.7)

decision_tree_model = DecisionTree(_max_depth=4, _min_splits=30)
decision_tree_model.train(iris_train, iris_train_labels)

prediction = decision_tree_model.test(iris_test, iris_test_labels)

"""from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()
feature = iris.data[:,:4]
label = iris.target

X_train, X_test, y_train, y_test = train_test_split(feature, label, random_state= 42)

print(X_train)"""
