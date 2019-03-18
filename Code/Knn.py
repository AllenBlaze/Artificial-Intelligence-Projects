"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement
	* predict 	: pour prédire la classe d'un exemple donné
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes test, predict et test de votre code.
"""


import numpy as np
import math


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn:

	def __init__(self, **kwargs):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""

		self.k = kwargs["k"]

	# Fonction pour alculer de la distance euclidienne entre un point_1 et un point_2
	def euclidean_distance(self, point_1, point_2):
			distance = 0
			for x in range(len(point_1)):
				distance += pow((point_1[x] - point_2[x]), 2)
			#print(math.sqrt(distance))
			return math.sqrt(distance)

	# Fonction pour obtenir les index des plus proches voisins d'un point en fonction de la distance qui les sépare
	def get_neighbors(self, train_data, test_data, k):
		distances = []
		for index in range(len(train_data)):
			dist = self.euclidean_distance(test_data, train_data[index])
			distances.append(dist)
		# Ordonner les distances en ordre croissant mais retourner uniquement les indexes
		distances_index_sorted = sorted(range(len(distances)), key=lambda k: distances[k])
		neighbors = []
		# Prendre les k plus proches voisins et les retourner
		for index in range(k):
			neighbors.append(distances_index_sorted[index])
		return neighbors

	# Fonction pour mettre à jour la matrix de conufusion
	def update_confusion_matrix(self, predicted_label, real_label, confusion_matrix):
		confusion_matrix[predicted_label, real_label] += 1

	# Fonction pour obtenir la precision
	def get_precision(self, label, confusion_matrix):
		col = confusion_matrix[:, label]
		return round(confusion_matrix[label, label] / col.sum() * 100, 2)

	# Fonction pour obtenir le recall
	def get_recall(self, label, confusion_matrix):
		row = confusion_matrix[label, :]
		return round(confusion_matrix[label, label] / row.sum() * 100, 2)


	def train(self, train, train_labels): #vous pouvez rajouter d'autres attribus au besoin
		"""
		c'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		
		
		------------
		Après avoir fait l'entrainement, faites maintenant le test sur 
		les données d'entrainement
		IMPORTANT : 
		Vous devez afficher ici avec la commande print() de python,
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)
		
		Bien entendu ces tests doivent etre faits sur les données d'entrainement
		nous allons faire d'autres tests sur les données de test dans la méthode test()
		"""
		self.train = train
		self.train_labels = train_labels
		# Obtenir le nom de classe disctinct dans le dataset
		x = np.unique(self.train_labels)
		y = len(x)
		train_confusion_matrix = np.zeros((y, y), dtype=int)
		for item in range(len(train)):
			prediction_result = self.predict(train[item], train_labels[item])
			# Mettre la matrix en convertissant les labels originaux en entier car ils sont decimaux à l'origine
			self.update_confusion_matrix(prediction_result[1], int(prediction_result[2]), train_confusion_matrix)
		print('Matrix de confusion pour l`entrainement')
		print(train_confusion_matrix)

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""

		# Obtenir les index des k plus proches voisins
		neighbors_index = self.get_neighbors(self.train, exemple, self.k)
		# Obtenir le nom de classe disctinct dans le dataset
		x = np.unique(self.train_labels)
		y = len(x)
		# Initialiser les compteurs de vote à 0
		vote_counter = np.zeros(y, dtype=int)
		neighbors_labels = self.train_labels[neighbors_index].astype(int)
		# Ajouter 1 point pour chaque label present dans la liste des voisins les plus proches
		for label_index in range(len(neighbors_labels)):
			vote_counter[neighbors_labels[label_index]] += 1
		# Selectionner le label majoritaire
		max_args_label = np.argmax(vote_counter)
		predicted_label = max_args_label
		prediction_is_correct = predicted_label == label
		result = [prediction_is_correct, predicted_label, label[0]]
		return result


	def test(self, test, test_labels):
		"""
		c'est la méthode qui va tester votre modèle sur les données de test
		l'argument test est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		test_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		Faites le test sur les données de test, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)
		
		Bien entendu ces tests doivent etre faits sur les données de test seulement
		
		"""

		self.test = test
		self.test_labels = test_labels
		# Obtenir le nom de classe disctinct dans le dataset
		x = np.unique(self.test_labels)
		y = len(x)
		test_confusion_matrix = np.zeros((y, y), dtype=int)
		for item in range(len(test)):
			prediction_result = self.predict(test[item], test_labels[item])
			# Mettre à jour la matrix en convertissant les labels originaux en entier car ils sont decimaux à l'origine
			self.update_confusion_matrix(prediction_result[1], int(prediction_result[2]), test_confusion_matrix)
		#presentation de la matrix de confusion
		print('Matrix de confusion pour le test')
		print(test_confusion_matrix)
		#Calcul de precision et du recall
		label_nummber = np.unique(test_labels)
		label_size = np.array(label_nummber, dtype= int)
		for actual_label in label_size:
			precision = self.get_precision(actual_label, test_confusion_matrix)
			presicion_output = f"precision of label {actual_label} is {precision}"
			recall = self.get_recall(actual_label, test_confusion_matrix)
			recall_output = f"recall of label {actual_label} is {recall}"
			print(presicion_output)
			print(recall_output)
			print('--------------------')

	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.