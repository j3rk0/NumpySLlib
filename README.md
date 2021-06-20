# NumpySLlib

A collection of hand-made statistical learning algorithms built with numpy.  
this includes for:
- feature selection 
- feature manipulation
- validation & valutation
- clustering & classification
___

## Provided Functions

### split

* *train_test_split(X, y, n_train)*: split a dataset in train and test. n_train is the percentage of data in train 
  set (es 0.75 = 75%). return in form x_train, y_train, x_test, y_test
* *k_fold(X,y,k)*: return a list of k subsets of X and one of subsets of y. the size of each folds differ at most of size 1 from the others

### Cross-validation:

* *cross_valid_F1_macro(Data, labels, k, model)*: f1 cross 
  validation macro average
* *cross_valid_A_macro(Data, labels, k, model)*: accuracy cross
  validation macro average
* *cross_valid_F1_micro(Data, labels, k, model)*: f1 cross 
  validation micro average
* *cross_valid_A_micro(Data, labels, k, model)*: accuracy cross
  validation micro average  
  
#### parameters:

- Data: numpy bi-dimensional array
- label: numpy array
- k: for k-fold cross validation
- model: a machine learning models that expose *fit* and *predict*
    methods (ex. sklearn models)


### feature manipulation

these function transform the feature with index *column_index* of 
the data provided with the parameter *to_normalize*.
Values like min, max or mean of the feature must be provided


* *normalize_transform(to_normalize, column_index, max_value, min_value)*:
  normalize in the span [0-1]
* *center_transform(to_normalize, column_index, mean_value)*: 
  center the feature around the mean
* *sig_transform(to_normalize, column_index, b)*:
  sigmoid transformation
* *log_transform(to_normalize, column_index, b)*:
  logaritmic transformation
* *clip_transform(to_normalize, column_index, b, up_or_down="up")*:
    clip the data. Via the *up_or_down* parameter you can choose 
  how the data should be cliped. you can clip both "up" 
  ( upper bounds the data to b), "down" (lower bounds data to b) or
  "up_down" (both upper and lower bounds)

### feature selection

* *backward_feature_elimination(data, model, labels, min_features)* :
  backward eliminate feature untill reached *min_features* (if setted), or
  if reached a minimum error (no feature can be eliminated without losing accuracy)
* *forward_feature_insertion(data, model, labels,  k)* : select the top
 *k* features using forward insertion


### clustering

#### clustering algorithms

* *k_means(data, k, n_iter)*: the most known clustering algorithm, it took a bidimensional
  numpy array *data* and a number *k* and it cluster data into k subset.
  *n_iter* parameter can be used to set the number of iteration of centroids
  refresh. Returns the labeled data.
* Dendrogram (Agglomerative): this algorithm is provided as a class. The constructor takes
  the data point of the cluster. Dendrogram implements method to grow, cut and flat dendrogram:
  
  * *grow_a_level(method, k)*: grow the dendrogram of a level,
  method can be set as: *"min"* (min linkage), *"avg"* (average linkage), 
    *"max"* (default, complete linkage). k optional parameter is set to be the max number of cluster
  * *grow_all(method)*: grow the full dendrogram.
  * *grow_k(k, method)*: grow the dendrogram untill k cluster are reached
  * *cut(level)*: cut the dendrogram at *level* and return the labeled data.
  * *get_k_cluster(k)*: cut the dendrogram to have *k* clusters and returns the labeled dataset

#### clustering valutation & validation

* *rss(X, k, labels)*: calculate the sum of the distances between the data *X*
   and the centroids of the *k* clusters. labels is an array of the cluster labels
* *purity_score(clust, labels)*: evalutate clustering of labeled data.
  *clust* are the cluster labels and *labels* are the ground truth labels
* *rand_index(clust, labels)*: same as purity score but with randindex measure
* *silhouette(data, label)*: calculate the silhouette score of a cluster, data is the
dataset and label is an array with cluster labels

### Decision Tree Classifier

this algorithm is provided as a class.
there are provided 3 main methods

* *fit(data, labels, level)* : fit the tree, optional parameter level it's to set max tree height
* *predict(data)* : return a numpy array of predictions
* *print_tree()* : print the structure of the decision tree

constructor:
* *DecisionTree(errfun=gini, max_height=None, min_err=0)* : errfun could be entropy or gini, or a user defined error function. default is gini.
  the other  two parameters are for pruning.

### Random Forest Classifier

this algorithm is provided as a class.
there are 2 main methods, basically the same as DecisionTree.
Random forests have a lower risk of overfitting.
the training of the trees of the forest could be.

constructor:
* *RandomForest(n_trees, max_height=2, min_features, min_data, errfun, n_processes=1)*:
  n_trees and max_height are the parameters of each tree. min_features and min_data are the minimum amount
  of data/features given to each trees. default are the half of the shape of the dataset.
  If errfun is not set each tree would hav a random errfun between gini and entropy.
  n_process is for the parallel grow