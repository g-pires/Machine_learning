from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf, out_file='essai.dot')
clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf)

#cross validation
print(cross_val_score(clf, iris.data, iris.target, cv=10))


#matrice de confusion
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
classifieur = tree.DecisionTreeClassifier()
y_pred = classifieur.fit(X_train, y_train).predict(X_test)
cm =confusion_matrix(y_test, y_pred)
#print(cm)

#matrice de confusion2
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33)
classifieur = tree.DecisionTreeClassifier()
y_pred = classifieur.fit(X_train, y_train).predict(X_test)
cm =confusion_matrix(y_test, y_pred)
#print(cm)
print('Arbre:',accuracy_score(y_test, y_pred))
print('Arbre_test:', classifieur.score(X_test, y_test))
#partie2
x2=iris.data
y2=iris.target

#K-means
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.33, random_state=42)
model = KMeans(n_clusters=2)
model.fit(X_train2)
model_X_train2 = model.labels_
model_X_train_pred2 = model.predict(X_test2)


def accuracy_score2():
    cnt = 0
    liste = []
    for j in range(100):
        cnt = 0
        cnt_2 = 0
        X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.33)
        model = KMeans(n_clusters=2)
        model.fit(X_train2)
        model_X_train2 = model.labels_
        model_X_train_pred2 = model.predict(X_test2)
        for i in range(len(model_X_train_pred2)):
            if model_X_train_pred2[i] == y_test2[i]:
                cnt += 1
        cnt_2 = (cnt/len(y_test2))*100
        liste.append(cnt_2)
    print('Prédiction : ', (sum(liste)/len(liste)))

print('Kmeans:', accuracy_score(y_test2, model_X_train_pred2))
print('Kmeans_test:', model.score(X_test2, y_test2))
#KMeans=0.68

#K-nearest Neighbors
iris = load_iris()
x3=iris.data
y3=iris.target

X_train3, X_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.33)

model3 = KNeighborsClassifier(n_neighbors=2)
model3.fit(X_train3, y_train3)
model3_X_train_pred = model3.predict(X_test3)

print('K-NN:', accuracy_score(y_test3, model3_X_train_pred))
print('K-NN_test:', model3.score(X_test3, y_test3))
#print(confusion_matrix(y_test3, model3_X_train_pred))
#K-nn = 0.88-0.98

