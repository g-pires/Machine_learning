from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#1
mnist = fetch_mldata('MNIST original')
mnist

#2
mnist.target.shape #pour avoir le nombre d'instance
mnist.data.shape

#3
sample = np.random.randint(70000, size=5000)
data=mnist.data[sample]
target=mnist.target[sample]

#4
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
#5
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
model_X_train_pred = model.predict(X_test)
print(accuracy_score(y_test, model_X_train_pred))

#6
liste = []
for i in range(2, 15):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    model_X_train_pred = model.predict(X_test)
    liste.append(accuracy_score(y_test, model_X_train_pred))
plt.plot(range(2, 15), liste, 'o-')
plt.show()

images = X_test.reshape((-1, 28, 28))

select = np.random.randint(images.shape[0], size=12)

for i, v in enumerate(select):
    plt.subplot(3, 4, i+1)
    plt.axis('off')
    plt.imshow(images[v], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction:%i'% model_X_train_pred[v])

#7
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf = clf.fit(data, target)
tree.export_graphviz(clf, out_file='essai2.dot')
clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf = clf.fit(data, target)
tree.export_graphviz(clf)

#cross validation
print(cross_val_score(clf, data, target, cv=10))

#matrice de confusion
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
classifieur = tree.DecisionTreeClassifier()
y_pred = classifieur.fit(X_train, y_train).predict(X_test)
cm =confusion_matrix(y_test, y_pred)
print(cm)