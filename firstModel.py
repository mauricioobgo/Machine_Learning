from sklearn.neighbors import KNeighborsClassifier

#Labels 
X = [[0], [1], [2], [3]]
#Data according to labels to train
y = [0, 0, 1, 1]

#n_neighbors, me dice cuantos puntos cercanos debería tener para clasificarlo en una categoría
neigh = KNeighborsClassifier(n_neighbors=3)

#Entrenamiento de modelo
neigh.fit(X, y) 


#Predicción de modelo, con un dato nuevo
print(neigh.predict_proba([[0.9]]))