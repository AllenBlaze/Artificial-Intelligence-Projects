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

	# Fonction pour obtenir la matrix de confusion
	#def get_confusion_matrix(self, ):

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

	# Fonction pour obtenir l'accuracy
	def get_accuracy(self, confusion_matrix):
		diagonal = np.diag(confusion_matrix)
		diagonal_sum = np.sum(diagonal)
		matrix_sum = confusion_matrix.sum()
		return round(diagonal_sum / matrix_sum * 100, 2)

	# Fonction pour obtenir les folds d'un data_set
	def get_data_k_fold(self, dataset, valeur_l):
		data_fold = np.array_split(dataset, valeur_l)
		return data_fold

	def get_label_k_fold(self, label, valeur_l):
		label_fold = np.array_split(label, valeur_l)
		return  label_fold

	def get_precision_total(self, confusion_matrix):
		rows, columns = confusion_matrix.shape
		precision_total = 0
		for label in range(rows):
			precision_total += self.get_precision(label, confusion_matrix)
		return round(precision_total / rows, 2)

	# Fonction pour obtenir le recall total

	def get_recall_total(self, confusion_matrix):
		rows, columns = confusion_matrix.shape
		precision_total = 0
		for label in range(columns):
			precision_total += self.get_recall(label, confusion_matrix)
		return round(precision_total / columns, 2)



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
		print('PERFORMANCE SUR TRAIN')
		print(train_confusion_matrix)
		# Calcul de precision et du recall
		label_nummber = np.unique(train_labels)
		label_size = np.array(label_nummber, dtype=int)
		for actual_label in label_size:
			accuracy = self.get_accuracy(train_confusion_matrix)
			precision = self.get_precision(actual_label, train_confusion_matrix)
			presicion_output = f"precision of label {actual_label} is {precision}"
			recall = self.get_recall(actual_label, train_confusion_matrix)
			recall_output = f"recall of label {actual_label} is {recall}"
			print(presicion_output)
			print(recall_output)
		accuracy = self.get_accuracy(train_confusion_matrix)
		precision_total = self.get_precision_total(train_confusion_matrix)
		recall_total = self.get_recall_total(train_confusion_matrix)
		print('accuracy for train = ', accuracy)
		print('precision total for train = ', precision_total)
		print('recall total for train = ', recall_total)
		print('')

		#self.get_optimal_k(self.k, 5)

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


		# Obtenir le nom de classe disctinct dans le dataset
		x = np.unique(self.train_labels)
		y = len(x)

		test_confusion_matrix = np.zeros((y, y), dtype=int)
		for item in range(len(test)):
			prediction_result = self.predict(test[item], test_labels[item])
			# Mettre à jour la matrix en convertissant les labels originaux en entier car ils sont decimaux à l'origine
			self.update_confusion_matrix(prediction_result[1], int(prediction_result[2]), test_confusion_matrix)
		#presentation de la matrix de confusion
		#print('PERFORMANCE SUR TEST')
		print(test_confusion_matrix)
		#Calcul de precision et du recall
		label_nummber = np.unique(test_labels)
		label_size = np.array(label_nummber, dtype= int)
		'''
		for actual_label in label_size:
			accuracy = self.get_accuracy(test_confusion_matrix)
			precision = self.get_precision(actual_label, test_confusion_matrix)
			presicion_output = f"precision of label {actual_label} is {precision}"
			recall = self.get_recall(actual_label, test_confusion_matrix)
			recall_output = f"recall of label {actual_label} is {recall}"
			print(presicion_output)
			print(recall_output)
		'''
		accuracy = self.get_accuracy(test_confusion_matrix)
		precision_total = self.get_precision_total(test_confusion_matrix)
		recall_total = self.get_recall_total(test_confusion_matrix)
		print('accuracy total  = ', accuracy)
		print('precision total  = ', precision_total)
		print('recall total  = ', recall_total)
		print('')
			#print('--------------------')

		#self.cross_validation(test, test_labels, 5)
		return  accuracy

	def get_optimal_k(self, kmin,kmax ):

		k_value = self.k
		L = 10
		#old_k = self.k
		train_o = self.train
		train_labels_o = self.train_labels

		kmax = min(kmax, self.train.shape[0])
		kmin = max(1, kmin)

		k_list = range(kmin, kmax + 1)


		k_performances = np.zeros(len(k_list))

		for i, k in enumerate(k_list):
			print('k =', k)
			self.k = k
			k_performances[i] = self.cross_validation(L)

			self.train = train_o
			self.train_labels = train_labels_o

		self.k = k_value



		optimal_k = k_list[np.argmax(k_performances)]


		self.k = optimal_k
		print('-------PERFORMANCES GENERALES SUR LES DONNÉES DE TEST À PRENDRE EN COMPTE---------')
		print('The best k is : ', optimal_k)
		accuracy_output = f"accuracy of k = {optimal_k} is {max(k_performances)}"
		print(accuracy_output)
		return optimal_k

	def init_train_data(self, train, train_labels):
		self.train = train
		self.train_labels = train_labels

	def cross_validation(self, L = 10):
		train_o = self.train
		train_labels_o = self.train_labels

		nb_obs = (len(train_o) // L)

		list_L_fold_accuracy = np.zeros(L)


		for index in range(L):
			print("fold ",(index + 1))

			df = np.copy(self.train)
			df_lab = np.copy(self.train_labels)

			valid_start_ind = (index * L)
			valid_end_ind = ((index +1) * L)

			valid = df[valid_start_ind:valid_end_ind, :]
			valid_lab = df_lab[valid_start_ind:valid_end_ind]
			self.train = np.concatenate([df[:valid_start_ind, :], df[valid_end_ind:, :]])
			self.train_labels = np.concatenate([df_lab[:valid_start_ind], df_lab[valid_end_ind:]])


			x = self.test(valid, valid_lab)

			#x = self.get_it(valid, valid_lab)
			#self.display(self.test_confusion_matrix)
			#print("Accuracy is : ", x)
			np.put(list_L_fold_accuracy, [index], x)
			self.train = train_o
			self.train_labels = train_labels_o


		return list_L_fold_accuracy.mean()

		"""
			train_k_fold = self.get_data_k_fold(train, L)
			label_k_fold = self.get_label_k_fold(train_labels, L)

			validation_data = train_k_fold[index]
			validation_data_label = label_k_fold[index]

			del train_k_fold[index]
			del label_k_fold[index]


			train_data = np.concatenate(train_k_fold, axis=0)
			train_data_label = np.concatenate(label_k_fold, axis=0)
			accuracy = self.test(validation_data, validation_data_label)
			print("Accuracy is : ", accuracy)
			list_L_fold_accuracy.append(accuracy)

			self.init_train_data(train, train_labels)

		L_fold_sum = sum(list_L_fold_accuracy)
		L_fold_length = len(list_L_fold_accuracy)
		L_fold_mean = L_fold_sum / L_fold_length

		return L_fold_mean
		"""

	def get_it(self, test, test_labels):
		d = 1

		x = np.unique(self.train_labels)
		y = len(x)

		self.test_confusion_matrix = np.zeros((y, y), dtype=int)
		for item in range(len(test)):
			prediction_result = self.predict(test[item], test_labels[item])
			# Mettre à jour la matrix en convertissant les labels originaux en entier car ils sont decimaux à l'origine
			self.update_confusion_matrix(prediction_result[1], int(prediction_result[2]), self.test_confusion_matrix)

		label_nummber = np.unique(test_labels)
		label_size = np.array(label_nummber, dtype=int)
		for actual_label in label_size:
			accuracy = self.get_accuracy(self.test_confusion_matrix)
			precision = self.get_precision(actual_label, self.test_confusion_matrix)
			presicion_output = f"precision of label {actual_label} is {precision}"
			recall = self.get_recall(actual_label, self.test_confusion_matrix)
			recall_output = f"recall of label {actual_label} is {recall}"

		return accuracy

	def display(self, a):


		print(self.test_confusion_matrix)

	def set_nbNeighbors(self, k):
		self.k = k




		'''
		#k_min = self.k
		#k_max = 5
		train_k_fold = self.get_data_k_fold(train, L)
		label_k_fold = self.get_label_k_fold(train_labels, L)
		for index in range(L):


			test_data = train_k_fold[index]
			del train_k_fold[index]
			train_data = train_k_fold
			train_data = np.concatenate(train_data, axis=0)


			test_data_label = label_k_fold[index]
			del label_k_fold[index]
			train_data_label = label_k_fold
			train_data_label = np.concatenate(train_data_label, axis=0)

			#for k in range(k_max):

			#	print('test data is : ', test_data)
			#	print('train data is : ', train_data)
			print('------------- le train -------------- ')

			#self.train(train_data, train_data_label)

			train_k_fold = self.get_data_k_fold(train, L)
			label_k_fold = self.get_label_k_fold(train_labels, L)
			
		'''

	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.