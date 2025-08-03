#  K-Nearest Neighbors — Implémentation from Scratch en Python

**Auteur : Max89222**

---

Ce projet vise à recréer le modèle K-Nearest Neighbors from scratch sans utiliser le modèle KNeighborsClassifier de sklearn ou de toute autre librairie

## 1) Structure et utilité des fichiers

`main.py`: code source de notre projet 

## 2) Fonctionnement et architecture du code

Mon code utilise le concept de la Programmation Orientée Objet (POO) et contient une classe (KNeighborsClassifier) qui contient elle même plusieurs méthodes : 

- fit (pour entraîner notre modèle)
- predict (pour effectuer des prédictions)
- score (pour calculer l'accuracy de notre modèle)
  
## 3) Technologies utilisées

- python (POO notamment)
- matplotlib (affichage des graphiques)
- pandas / numpy (gestion de bases de données et manipulation de matrices)
- sklearn (UNIQUEMENT pour la fonction train_test_split et make_classification pour importer notre dataset)
- tqdm (suivre l'avancée d'une boucle for)

## 4) Résultats et métriques

Le score obtenu dépend de la configuration du dataset.
Pour une classification à 3 classes avec 10 000 échantillons, l'accuracy obtenu est de  0.947 mais libre à vous de paramétrer le dataset comme vous le souhaitez

## 5) Installation

1. Installer Git (si ce n’est pas déjà fait) :
   
`brew install git`

2. Cloner le dépôt :

`git clone <clé_ssh>`
`cd <nom_du_dossier>`

3. Installer les dépendances :

`pip3 install pandas scikit-learn matplotlib numpy tqdm`

4. Entraîner le modèle :

Ouvrir main.py dans un éditeur de code et l'exécuter. Une barre de progression va alors s'afficher.
Une fois l'entraînement du modèle terminé, l'accuracy de notre modèle ainsi que les paramètres optimaux s'afficheront dans la console

## 6) Idées d'amélioration et contributions
De nombreux points restent à améliorer notamment dans l'optimisation des hyperparamètres (type de distance etc) et dans la gestion des votes en cas d'égalité. 
N'hésitez pas à contribuer à ce projet ou à proposer des idées d'optimisation !

## 7) Concepts et formules mathématiques utilisés :

1) La norme euclidienne d’un vecteur $v = (v_1, v_2, ..., v_n)$ est donnée par :

$$
\|v\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}
$$


2) L'accuracy est définie comme :

$$
\text{Accuracy} = \frac{\text{Nombre de bonnes prédictions}}{\text{Nombre total de prédictions}} = \frac{TP + TN}{TP + TN + FP + FN}
$$

3) une bonne compréhension des matrices / vecteurs (et du broadcasting) est également nécessaire pour comprendre pleinement le fonctionnement du modèle

