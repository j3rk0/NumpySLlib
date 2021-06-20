import numpy as np
from sklearn.datasets import load_wine
from preprocessing.split import train_test_split
from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
from classifiers.knn import KNN
import valutation as val
from preprocessing.features_enginering import normalize_dataset
from sklearn.datasets import load_boston


def sqrderr(res, labels):
    err = 0
    for i in range(len(labels)):
        err += np.absolute(res[i] - labels[i])
    return err / len(labels)


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

print("knn err: ", knn_err, " knn weig:", knn_w_err, " rf weighted: ", err_w, " rf:", err, " tree:", err_tree)

# %% CLASSIFICATION

X, y = load_wine(return_X_y=True)

normalize_dataset(X)
x_train, y_train, x_test, y_test = train_test_split(X, y, .8)

## %%

kclass = KNN(5, method="inverse")
kclass.fit(x_train, y_train, )

## %%
res = kclass.predict(x_test)
knn_acc = val.A_micro_average(y_test, res)

## %%
tree = DecisionTree()
tree.fit(x_train, y_train)
forest = RandomForest(50, 2, weighted=True, n_processes=4)
forest.fit(x_train, y_train)

## %%

forest_pred = forest.predict(x_test)
tree_pred = tree.predict(x_test)
forest_acc = val.A_micro_average(y_test, forest_pred)
tree_acc = val.A_micro_average(y_test, tree_pred)

print("forest acc:", forest_acc, " tree acc:", tree_acc, " knn act: ", knn_acc)
