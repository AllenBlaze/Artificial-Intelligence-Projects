import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
from Knn import Knn # importer la classe du Knn
from BayesNaif import  BayesNaif
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initializer vos paramètres

k = 1



# Initializer/instanciez vos classifieurs avec leurs paramètres


knn = Knn(k=k)
bayesNaif_iris = BayesNaif()
bayesNaif_vote = BayesNaif()
bayesNaif_monks = BayesNaif()

# Charger/lire les datasets

iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(0.7)

congressional_train, congressional_train_labels, congressional_test, \
    congressional_test_labels = load_datasets.load_congressional_dataset(0.7)

monks_train, monks_train_labels, monks_test, monks_test_labels = load_datasets.load_monks_dataset(3)

# Entrainez votre classifieur

#knn.train(train, train_labels)
print('--------------------')
print('IRIS DATASET TRAIN')
print('--------------------')
bayesNaif_iris.train(iris_train, iris_train_labels)

print('--------------------')
print('CONGRESSIONAL DATASET TRAIN')
print('--------------------')
bayesNaif_vote.train(congressional_train, congressional_train_labels)

print('--------------------')
print('MONKS DATASET TRAIN')
print('--------------------')
bayesNaif_monks.train(monks_train, monks_train_labels)


# Tester votre classifieur

#knn.test(test, test_labels)
print('--------------------')
print('IRIS DATASET TEST')
print('--------------------')
bayesNaif_iris.test(iris_test, iris_test_labels)

print('--------------------')
print('CONGRESSIONAL DATASET TEST')
print('--------------------')
bayesNaif_vote.test(congressional_test, congressional_test_labels)

print('--------------------')
print('MONKS DATASET TEST')
print('--------------------')
bayesNaif_monks.test(monks_test, monks_test_labels)