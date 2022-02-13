from sklearn.cluster import KMeans
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
x=iris.data
y=iris.target

#K-means
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = KMeans(n_clusters=3)
model.fit(X_train)
model_X_train = model.labels_
model_X_train_pred = model.predict(X_test)


def accuracy_score2():
    cnt = 0
    liste = []
    for j in range(100):
        cnt = 0
        cnt_2 = 0
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        model = KMeans(n_clusters=3)
        model.fit(X_train)
        model_X_train = model.labels_
        model_X_train_pred = model.predict(X_test)
        for i in range(len(model_X_train_pred)):
            if model_X_train_pred[i] == y_test[i]:
                cnt += 1
        cnt_2 = (cnt/len(y_test))*100
        liste.append(cnt_2)
    print('Prédiction : ', liste)


#K-Nearest Neighbors
wine = pd.read_csv('winequality-white.csv', sep=';')
l, f = wine.shape
x2 = np.array(wine.ix[:, 0:f-1])
y2 = np.array(wine['quality'])

X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.33)

model2 = KNeighborsClassifier(n_neighbors=2)
model2.fit(X_train2, y_train2)
model2_X_train_pred = model2.predict(X_test2)

print(accuracy_score(y_test2, model2_X_train_pred))

print(confusion_matrix(y_test2, model2_X_train_pred))
