import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("KNNPoints.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

print(X)
print(y)

classifier = KNeighborsClassifier(n_neighbors=3) #by default it takes weight='distance'
classifier.fit(X, y)

X_test = np.array([6, 6])  #we have find
y_pred = classifier.predict([X_test])
print('General KNN', y_pred)


#Ditance weighted KNN
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
classifier.fit(X, y)

X_test = np.array([6, 2])
y_pred = classifier.predict([X_test])
print('Distance Weighted KNN', y_pred)



 
