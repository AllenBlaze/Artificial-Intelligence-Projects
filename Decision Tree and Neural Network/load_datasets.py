import numpy as np
import random
#import pandas as pd


def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque
        ligne dans cette matrice représente un exemple (ou instance) de test.

        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché
    f = open('datasets/bezdekIris.data', 'r')

    # TODO : le code ici pour lire le dataset
    # Lecture du dataset iris et transforamtion des labels en nombre
    labels = [b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica']
    conv = lambda x: labels.index(x)
    iris_dataframe = np.loadtxt('datasets/bezdekIris.data', delimiter=",", converters={4: conv})
    # Mixage des records du dataset
    np.random.shuffle(iris_dataframe)
    # Diviser les données en 4 ensembles
    iris_dataframe_size = iris_dataframe.shape
    # Spliter le dataset en train_set et test_set
    train_size = (int)(train_ratio * iris_dataframe_size[0])
    train = iris_dataframe[0:train_size, 0:4]
    train_labels = iris_dataframe[0:train_size, -1:]
    test = iris_dataframe[train_size:, 0:4]
    test_labels = iris_dataframe[train_size:, -1:]

    # REMARQUE très importante :
    # remarquez bien comment les exemples sont ordonnés dans
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)


def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque
        ligne dans cette matrice représente un exemple (ou instance) de test.

        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican': 0, 'democrat': 1,
                         'n': 0, 'y': 1, '?': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché
    f = open('datasets/house-votes-84.data', 'r')
    # Lecture du dataset iris et transforamtion des labels en nombre
    labels = [b'republican', b'democrat']
    conv = lambda x: labels.index(x)
    others_values = [b'n', b'y', b'?']
    conv2 = lambda x: others_values.index(x)
    congressional_dataframe = np.loadtxt('datasets/house-votes-84.data', delimiter=",",
                                         converters={0: conv, 1: conv2, 2: conv2, 3: conv2,
                                                     4: conv2, 5: conv2, 6: conv2, 7: conv2, 8: conv2, 9: conv2,
                                                     10: conv2, 11: conv2, 12: conv2,
                                                     13: conv2, 14: conv2, 15: conv2, 16: conv2})

    # Mixage des records du dataset
    np.random.shuffle(congressional_dataframe)
    # Diviser les données en 4 ensembles
    congressional_dataframe_size = congressional_dataframe.shape
    # Spliter le dataset en train_set et test_set
    train_size = int(train_ratio * congressional_dataframe_size[0])
    train = congressional_dataframe[0:train_size, 1:17]
    train_labels = congressional_dataframe[0:train_size, 0:1]
    test = congressional_dataframe[train_size:, 1:17]
    test_labels = congressional_dataframe[train_size:, 0:1]

    # TODO : le code ici pour lire le dataset

    # La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)


def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks

    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    # TODO : votre code ici, vous devez lire les fichiers .train et .test selon l'argument numero_dataset

    f1 = open('datasets/monks-' + str(numero_dataset) + '.train', 'r')
    f2 = open('datasets/monks-' + str(numero_dataset) + '.test', 'r')

    monk_dataset_train = np.loadtxt(f1, delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7))
    monk_dataset_test = np.loadtxt(f2, delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7))

    train = monk_dataset_train[0:, 1:6]
    train_labels = monk_dataset_train[0:, 0:1]

    test = monk_dataset_test[0:, 1: 6]
    test_labels = monk_dataset_test[0:, 0:1]

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)


if __name__ == '__main__':
    load_monks_dataset(3)