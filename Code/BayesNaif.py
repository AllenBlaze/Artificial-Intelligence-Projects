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
import  math


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class BayesNaif:  # nom de la class à changer

    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """

    # Fonction pour mettre à jour la matrix de conufusion
    def update_confusion_matrix(self, predicted_label, real_label, confusion_matrix):
        confusion_matrix[predicted_label, real_label] += 1

    # Fonction pour obtenir la precision pour chaque label
    def get_precision_by_label(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return round(confusion_matrix[label, label] / col.sum() * 100, 2)

    # Fonction pour obtenir le recall pour chaque label
    def get_recall_by_label(self, label, confusion_matrix):
            row = confusion_matrix[label, :]
            return round(confusion_matrix[label, label] / row.sum() * 100, 2)

    # Fonction pour obtenir la precision total

    def get_precision_total(self, confusion_matrix):
        rows, columns = confusion_matrix.shape
        precision_total = 0
        for label in range(rows):
            precision_total += self.get_precision_by_label(label, confusion_matrix)
        return round(precision_total / rows, 2)

    # Fonction pour obtenir le recall total

    def get_recall_total(self, confusion_matrix):
        rows, columns = confusion_matrix.shape
        precision_total = 0
        for label in range(columns):
            precision_total += self.get_recall_by_label(label, confusion_matrix)
        return round(precision_total / columns, 2)

    #Fonction pour obtenir l'accuracy
    def get_accuracy(self, confusion_matrix):
        diagonal = np.diag(confusion_matrix)
        diagonal_sum = np.sum(diagonal)
        matrix_sum = confusion_matrix.sum()
        return  round(diagonal_sum / matrix_sum * 100, 2)

    def train(self, train, train_labels):  # vous pouvez rajouter d'autres attribus au besoin
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


        self.liste_label_unique = np.unique(self.train_labels)
        self.nombre_label_unique = len(self.liste_label_unique)

        print("TRAIN")
        self.liste_label_sorted = np.sort(self.liste_label_unique)


        #probabilite des label dans le jeu de donnees
        liste_probabilite_label = [self.probabilite_label(label) for label in self.liste_label_sorted]
        self.toute_probabilite_label = np.array(liste_probabilite_label)

        #Trouver la prediction d'un exemple donné
        self.train_size = train.shape

        # Initiliser la matrix de confusion
        train_confusion_matrix = np.zeros((self.nombre_label_unique, self.nombre_label_unique), dtype=int)

        # predir les labels de l'ensemble d'entrainement
        for index in range(self.train_size[0]):
            prediction_result = self.predict(self.train[index], self.train_labels[index])
            # Mettre la matrix en convertissant les labels originaux en entier car ils sont decimaux à l'origine
            self.update_confusion_matrix(prediction_result[0], int(prediction_result[1]), train_confusion_matrix)
        # presentation de la matrix de confusion
        print('Confusion matrix for training')
        print(train_confusion_matrix)
        print('')
        # Calcul de precision et du recall
        for actual_label in range(self.nombre_label_unique):
            precision = self.get_precision_by_label(actual_label, train_confusion_matrix)
            presicion_output = f"precision of label {actual_label} is {precision}"
            recall = self.get_recall_by_label(actual_label, train_confusion_matrix)
            recall_output = f"recall of label {actual_label} is {recall}"
            print(presicion_output)
            print(recall_output)
            print('')
        accuracy = self.get_accuracy(train_confusion_matrix)
        precision_total = self.get_precision_total(train_confusion_matrix)
        recall_total = self.get_recall_total(train_confusion_matrix)
        print('accuracy for train = ', accuracy)
        print('precision total for train = ', precision_total)
        print('recall total for train = ', recall_total)
        print('')


    def predict(self, exemple, label):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification

        """
        predicted_label = np.argmax(self.get_posterior(exemple))

        result = [predicted_label, label[0]]

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

        self.liste_label_unique = np.unique(self.train_labels)
        self.nombre_label_unique = len(self.liste_label_unique)

        print("TEST")

        # Initiliser la matrix de confusion
        test_confusion_matrix = np.zeros((self.nombre_label_unique, self.nombre_label_unique), dtype=int)

        # Trouver la prediction d'un exemple donné
        self.test_size = test.shape

        # predir les labels de l'ensemble de test
        for index in range(self.test_size[0]):
            prediction_result = self.predict(self.test[index], self.test_labels[index])
            # Mettre la matrix en convertissant les labels originaux en entier car ils sont decimaux à l'origine
            self.update_confusion_matrix(prediction_result[0], int(prediction_result[1]), test_confusion_matrix)

        # presentation de la matrix de confusion
        print('Confusion matrix for testing')
        print(test_confusion_matrix)
        print('')
        # Calcul de precision et du recall
        for actual_label in range(self.nombre_label_unique):
            precision = self.get_precision_by_label(actual_label, test_confusion_matrix)
            presicion_output = f"precision of label {actual_label} is {precision}"
            recall = self.get_recall_by_label(actual_label, test_confusion_matrix)
            recall_output = f"recall of label {actual_label} is {recall}"
            print(presicion_output)
            print(recall_output)
            print('')
        accuracy = self.get_accuracy(test_confusion_matrix)
        precision_total = self.get_precision_total(test_confusion_matrix)
        recall_total = self.get_recall_total(test_confusion_matrix)
        print('accuracy for test = ', accuracy)
        print('precision total for test = ', precision_total)
        print('recall total for test = ', recall_total)
        print('')




    def get_posterior(self, x):
        return self.toute_probabilite_label * self.get_likelihood_of_x(x)

    def get_likelihood_of_x(self, x):
        list_of_likelihood_values = []
        for label in self.liste_label_sorted:
            self.likelihood_by_x = np.prod(self.get_likelihood_of_each_x(x, label))
            list_of_likelihood_values.append(self.likelihood_by_x)
        return  np.array(list_of_likelihood_values)

    def get_likelihood_of_each_x(self, x, k):
        list_of_each_x_likehood  = []
        for index in range(len(x)):
            mean = self.get_mean(index, k)
            variance = self.get_variance(index, k)
            list_of_each_x_likehood.append(self.probability_of_exemple_knowing_k(x[index], mean, variance))
        return np.array(list_of_each_x_likehood)

    def probability_of_exemple_knowing_k(self, exemple, mean, variance):
        probability_exemple = 1 / (math.sqrt(2 * math.pi * variance)) * math.exp((-(exemple -mean)**2) / (2 * variance))
        return probability_exemple

    def probabilite_label(self, label):
        probability = np.sum(self.train_labels == label) / np.sum(self.train_labels)
        return probability

    def get_mean(self, i, k):
        label_indice = np.argwhere(self.train_labels == k)
        mean = np.mean(self.train[label_indice, i])
        return mean

    def get_variance(self, i, k):
        label_indice = np.argwhere(self.train_labels == k)
        variance = np.var(self.train[label_indice, i])
        return variance



# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.