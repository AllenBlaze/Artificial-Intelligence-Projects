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

k = 3



# Initializer/instanciez vos classifieurs avec leurs paramètres


knn_iris = Knn(k=k)
knn_vote = Knn(k=k)
knn_monks = Knn(k=k)
bayesNaif_iris = BayesNaif()
bayesNaif_vote = BayesNaif()
bayesNaif_monks = BayesNaif()

# Charger/lire les datasets

iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(0.7)

congressional_train, congressional_train_labels, congressional_test, \
    congressional_test_labels = load_datasets.load_congressional_dataset(0.7)

# Lecture des données pour monk = 1
monks_train_1, monks_train_labels_1, monks_tes_1, monks_test_labels_1 = load_datasets.load_monks_dataset(1)
# Lecture des données pour monk = 2
monks_trai_2, monks_train_labels_2, monks_test_2, monks_test_labels_2 = load_datasets.load_monks_dataset(2)
#Lecture des données pour monk = 3
monks_train_3, monks_train_labels_3, monks_test_3, monks_test_labels_3 = load_datasets.load_monks_dataset(3)

print('----------------------------')
print('----------------------------')
print('CLASSIFIEUR DE KNN')
print('----------------------------')
print('----------------------------')


# Entrainez votre classifieur KNN
print('--------------------')
print('IRIS DATASET TRAIN')
print('--------------------')
knn_iris.train(iris_train, iris_train_labels)



print('--------------------')
print('VOTE DATASET TRAIN')
print('--------------------')
knn_vote.train(congressional_train, congressional_train_labels)



print('--------------------')
print('MONK DATASET TRAIN')
print('--------------------')
knn_monks.train(monks_train_3, monks_train_labels_3)


# Testez votre classifieur KNN



print('--------------------')
print('IRIS DATASET TEST')
print('--------------------')
k_optimal = knn_iris.get_optimal_k(kmin=1, kmax=6)
#print('-------Performance générales sur les données de test---------')
knn_iris.set_nbNeighbors(k_optimal)
print('Running now on test data with k = ', k_optimal)
knn_iris.test(iris_test, iris_test_labels)



print('--------------------')
print('VOTE DATASET TEST')
print('--------------------')
k_optimal = knn_vote.get_optimal_k(kmin=1, kmax=6)
#print('-------Performance générales sur les données de test---------')
knn_vote.set_nbNeighbors(k_optimal)
print('Running now on test data with k = ', k_optimal)
knn_vote.test(congressional_test, congressional_test_labels)



print('--------------------')
print('MONKS DATASET TEST')
print('--------------------')
k_optimal = knn_monks.get_optimal_k(kmin=1, kmax=6)
#print('-------Performance générales sur les données de test---------')
knn_monks.set_nbNeighbors(k_optimal)
print('Running now on test data with k = ', k_optimal)
knn_monks.test(monks_test, monks_test_labels)




'''
print('----------------------------')
print('----------------------------')
print('CLASSIFIEUR NAIF DE BAYES')
print('----------------------------')
print('----------------------------')
# Entrainez votre classifieur NAIF BAYES



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

# Tester votre classifieur NAIF BAYES

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
'''

