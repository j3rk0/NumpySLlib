from sklearn.datasets import load_wine
from preprocessing.split import train_test_split
from classifiers.decision_tree import DecisionTree
from classifiers.random_forest import RandomForest
import valutation as val
from preprocessing.features_enginering import normalize_dataset

X, y = load_wine(return_X_y=True)

normalize_dataset(X)
x_train, y_train, x_test, y_test = train_test_split(X, y, .8)

# %%
from classifiers.knn import KNN

kclass = KNN(5, method="inverse")
kclass.fit(x_train, y_train,)


#%%
res = kclass.predict(x_test)
acc = val.A_micro_average(y_test,res)
cm = val.class_matrix(y_test, res)


#%%




#%%
tree = DecisionTree()
tree.fit(x_train, y_train)
forest = RandomForest(50, 2, n_processes=4)
forest.fit(x_train, y_train)


# %%

forest_pred = forest.predict(x_test)
tree_pred = tree.predict(x_test)
forest_acc = val.A_micro_average(y_test, forest_pred)
tree_acc = val.A_micro_average(y_test, tree_pred)

print("forest acc:", forest_acc, " tree acc:", tree_acc)

