from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

np.random.seed(0)

# création du dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=4,              
    n_clusters_per_class=1,
    random_state=42
)

# création de notre train_set et de notre test_set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class KNeighborsClassifier:     # création de notre classe KNeighborsClassifier
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):    # création de notre méthode fit pour entraîner notre modèle
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):      # création de notre méthode predict pour effectuer des prédictions
        self.X_test = X_test
        predictions = []
        for i in self.X_test:
            vecteur = self.X_train - i
            distances = pd.Series(np.linalg.norm(vecteur, axis=1))
            index = distances.nsmallest(self.n_neighbors).index
            predictions.append(pd.Series(self.y_train[index]).value_counts().idxmax())
       
        return np.array(predictions)

    def score(self, X_test, y_test):        # création de notre méthode score pour calculer l'accuracy
        predictions = self.predict(X_test)
        return (predictions == y_test).mean()


# création d'une boucle for pour trouver le paramètre n_neighbors optimal
k = np.arange(1, 10, 1)
accuracy = []
for i in tqdm(k):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    accuracy.append(model.score(X_test, y_test))

accuracy = np.array(accuracy)
best_param = k[accuracy.argmax()]   # enregistrement du paramètre optimal dans la variable best_param

model = KNeighborsClassifier(n_neighbors=best_param)       # création d'un object model
model.fit(X_train, y_train)         # entraînement de notre modèle
print('score final (accuracy) :', model.score(X_test, y_test))      # affichage du score de notre modèle
print('paramètre otptimal pour n_neighbors :', best_param)      

