# NumpySLlib

A collection of hand-made statistical learning models and tools built in plain numpy.  
this package includes:
- classification and regression models (knn, decision tree, svm and random forest)
- clustering models (kmeans and agglomerative clustering)  
- preprocessing functions (feature selection, feature manipulation, splitting)
- validation metrics for clustering, classification and regression and confusion matrixes
___

## Documentation:


### Models

#### Decision Tree

this class is a decision tree model for both regression or classification
there are  3 main methods provided:

* *fit(data, labels, level)* : fit the tree, optional parameter level it's to set max tree height
* *predict(data)* : return a numpy array of predictions
* *print_tree()* : print the structure of the decision tree

constructor:
* *DecisionTree(mode="classification", errfun=gini, max_height=None, min_err=0)* : mode can be set to "classification" 
  or "regression" errfun could be "entropy", "gini" or "mse" (only for regression). 
  the other  two parameters are for pruning. Currently only pre-pruning is supported.

#### Random Forest

this model is the bagging ensamble of decision tree. it implements both weighted and
majority voting. there are 2 main methods, basically the same as DecisionTree.
Random forests have a lower risk of overfitting.
the training of the trees of the forest could be done in multiprocessing.

constructor:
* *RandomForest(n_trees, max_height=2, min_err=.0, mode="classification", min_features=None, min_data=None,
                 errfun=None, n_processes=1, weighted=True)*:
  n_trees, mode, max_height and min_err are the parameters of each tree. min_features and min_data are the minimum amount
  of data/features given to each trees. default are the half of the shape of the dataset.
  If errfun is not set and mode is classification each tree would have a random errfun between gini and entropy.
  n_process is for the parallel grow. weighted indicates to use majority or weighted voting.

#### KNN

knn is a very common lazy SL model. it can be used for regression and classification.
majority or inverse voting is implemented.
fit and predict method are the same of the other models.
Constructor have the following signature:

 * *KNN(k, mode="classification", method="standard", smoothing=.1)* k is for the 
    number of neighbours, mode is classification or regression. method can be standard
   or inverse. for inverse voting you can also edit smoothing parameter
   
#### Linear SVM

Support vector machines are the most used linear predictors. here i used sgd algorithm
for the training.
There are two class for this implementation of Linear SVM. the BinaryLinearSVM support only
2 class classification. The LinearSVM indeed work for every number of class.
multiclass svm use one-vs-all approach.
The constructors take only two parameters: epoch number and learn rate.
The training of each svm in multiclass is done in multiprocessing

#### KMeans

the most known clustering algorithm. main 3 methods are provided:

* fit(X): fit the data provided as X and cluster them
* get_cluster_labels(): get the cluster labels
* fit_predict(X): fit the class and return the labels

constructor:

*   KMeans(k,n_iter): *k* is the number of cluster to divide data *n_iter* parameter can be 
    used to set the number of iteration of centroids refresh.
    

#### Dendrogram

this is an agglomerative clustering model The constructor takes
  the data point of the cluster. Dendrogram implements method to grow, cut and flat dendrogram:
  
  * *grow_a_level(method, k)*: grow the dendrogram of a level,
  method can be set as: *"min"* (min linkage), *"avg"* (average linkage), 
    *"max"* (default, complete linkage). k optional parameter is set to be the max number of cluster
  * *grow_all(method)*: grow the full dendrogram.
  * *grow_k(k, method)*: grow the dendrogram untill k cluster are reached
  * *cut(level)*: cut the dendrogram at *level* and return the labeled data.
  * *get_k_cluster(k)*: cut the dendrogram to have *k* clusters and returns the labeled dataset

### preprocessing


* feature manipulation:

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

* feature selection
    * *backward_feature_elimination(data, model, labels, min_features)* : backward eliminate feature untill reached *min_features* (if setted), or
if reached a minimum error (no feature can be eliminated without losing accuracy)
    * *forward_feature_insertion(data, model, labels,  k)* : select the top
 *k* features using forward insertion

* split

    * *train_test_split(X, y, n_train)*: split a dataset in train and test. n_train is the percentage of data in train 
  set (es 0.75 = 75%). return in form x_train, y_train, x_test, y_test
    * *k_fold(X,y,k)*: return a list of k subsets of X and one of subsets of y. the size of each folds differ at most of size 1 from the others


### validation

* classification:
  * confusion matrix:
    * *class_matrix(labels,predictions)* n X n confusion matrix
    * *confusion_matrix(labels, predictions)* binary confusion matrix
    * *confusion_matrix_multiclass(labels,predictions)* TP/FN/FP/TN matrix for multiclass
  * cross-validation: cros-validated f1 score and accuracy with macro and micro average
    model should be one of my models or a model sklearn-like wich implements fit and predict functions
    * *cross_valid_F1_macro(Data, labels, k, model)*
    * *cross_valid_A_macro(Data, labels, k, model)*
    * *cross_valid_F1_micro(Data, labels, k, model)*
    * *cross_valid_A_micro(Data, labels, k, model)*    
  * metrics:
    * *F1_micro_average(labels,predictions)*
    * *F1_macro_average(labels,predictions)*
    * *A_micro_average(labels,predictions)*
    * *F1_macro_average(labels,predictions)*
  


* clustering: clustering validation metrics
  * *rss(X,k,labels)*
  * *purity_score(clust,labels)*
  * *rand_index(clust,labels)*
  * *silhouette(data,label)*
    
    
* regression:
    * sqrderr(res,labels)
