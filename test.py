from sklearn.datasets import load_boston

from validation import classification as val
from models.decision_tree import DecisionTree
from models.knn import KNN
from models.random_forest import RandomForest
from preprocessing.features_enginering import normalize_dataset
from preprocessing.split import train_test_split
from validation.regression import sqrderr


# %%
X, y = load_boston(return_X_y=True)
normalize_dataset(X)
x_train, y_train, x_test, y_test = train_test_split(X, y, .8)

kclass = KNN(5, mode="regression")
kclass.fit(x_train, y_train)

res = kclass.predict(x_test)
knn_err = sqrderr(res, y_test)

kclass_w = KNN(5, mode="regression", method="weighted")
kclass_w.fit(x_train, y_train)

res = kclass_w.predict(x_test)
knn_w_err = sqrderr(res, y_test)

## %%

forest_w = RandomForest(mode="regression", errfun="mse", weighted=True, max_height=2, n_trees=20, n_processes=5)
forest_w.fit(x_train, y_train)

forest = RandomForest(mode="regression", errfun="mse", weighted=False, max_height=2, n_trees=20, n_processes=5)
forest.fit(x_train, y_train)

## %%

res = forest.predict(x_test)
err = sqrderr(res, y_test)

res = forest_w.predict(x_test)
err_w = sqrderr(res, y_test)

## %%

tree = DecisionTree(mode="regression", errfun="mse", max_height=3)
tree.fit(x_train, y_train)

## %%
res = tree.predict(x_test)
err_tree = sqrderr(res, y_test)

print(f"knn err:  {knn_err} knn weig: {knn_w_err} rf weighted: {err_w} rf: {err} tree: {err_tree}")

# %% CLASSIFICATION

import sklearn.ensemble as randfo
import sklearn.tree  as tr
import sklearn.neighbors as kn
from sklearn.datasets import load_breast_cancer

# X, y = load_wine(return_X_y=True)

X, y = load_breast_cancer(return_X_y=True)

normalize_dataset(X)
x_train, y_train, x_test, y_test = train_test_split(X, y, .8)

## %%

kclass = KNN(5, method="inverse")
kclass_sk = kn.KNeighborsClassifier(n_neighbors=5, weights="distance")
kclass.fit(x_train, y_train, )
kclass_sk.fit(x_train, y_train)

## %%
res = kclass.predict(x_test)
knn_acc = val.A_micro_average(y_test, res)
res = kclass_sk.predict(x_test)
knn_sk_acc = val.A_micro_average(y_test, res)
## %%
tree = DecisionTree()
tree_sk = tr.DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree_sk.fit(x_train, y_train)

forest = RandomForest(50, 2, weighted=True, n_processes=4)
forest_sk = randfo.RandomForestClassifier(n_estimators=50, max_depth=2)
forest.fit(x_train, y_train)
forest_sk.fit(x_train, y_train)

## %%

forest_pred = forest.predict(x_test)
forest_sk_pred = forest_sk.predict(x_test)
tree_pred = tree.predict(x_test)
tree_sk_pred = tree_sk.predict(x_test)
forest_acc = val.A_micro_average(y_test, forest_pred)
forest_sk_acc = val.A_micro_average(y_test, forest_sk_pred)
tree_acc = val.A_micro_average(y_test, tree_pred)
tree_sk_acc = val.A_micro_average(y_test, tree_sk_pred)

print("jerkosl:")
print(f"forest acc: {forest_acc}  tree acc: {tree_acc}  knn act:  {knn_acc}")
print("sklearn:")
print(f"forest acc: {forest_sk_acc} tree acc: {tree_sk_acc} knn act:  {knn_sk_acc}")

# %%
import numpy as np
import validation.classification as val
from preprocessing.features_enginering import normalize_dataset
from preprocessing.split import train_test_split
from sklearn.datasets import load_wine
from models.LinearSVM import LinearSVM

X, y = load_wine(return_X_y=True)

normalize_dataset(X)
x_train, y_train, x_test, y_test = train_test_split(X, y, .8)
# %%

svm = LinearSVM()
svm.fit(x_train, y_train)
#plt.scatter(x=err[:, 0], y=err[:, 1])
#plt.show()


# %%
pred = svm.predict(x_test)

# %%

acc = val.A_micro_average(y_test, pred)
print(f'svm accuracy: {acc}')