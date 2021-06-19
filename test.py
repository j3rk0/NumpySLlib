
from sklearn.datasets import load_wine
from split import train_test_split
from decision_tree import DecisionTree
import valutation as val

X, y = load_wine(return_X_y=True)
x_train, y_train, x_test, y_test = train_test_split(X, y, .8)

# %%
tree = DecisionTree()
tree.fit(x_train, y_train)

# %%

predictions = tree.predict(x_test)
accuracy = val.A_micro_average(y_test, predictions)
