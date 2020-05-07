###########################
# ALDA: hw2.R 
# Instructor: Dr. Thomas Price
# Group: G41
# @author: Yang Shi/yangatrue and Krishna Gadiraju/kgadira
# Srujana Rachakonda (srachak)
# Rajshree Jain (rjain27)
############################

require(caret)
require(rpart)

# ------- Part A -------

calculate_euclidean <- function(p, q) {
 
  combined_matrix <- rbind(p,q)
  euclidean_distance <- dist(combined_matrix, method="euclidean")
  return(euclidean_distance)
}

calculate_cosine <- function(p, q) {
 
  dot_product <- sqrt(sum(p^2)) * sqrt(sum(q^2))
  cosine_distance <- sum(p*q) / dot_product
  return(1-cosine_distance)
}
# gives predicted class per row 
get_class <- function (row, ctrain){
  # gets frequencies of classes 
  freq<-table(ctrain[row])
  # fetches the maximum repeating value
  res <-names(freq)[which (freq==max(freq))][1]
  return(res)
}
knn <- function(x_train, y_train, x_test, distance_method = 'cosine', k = 3){
 
  # if distance method is cosine we call calculate_cosine method else we call calculate_euclidean
  if (distance_method == 'cosine')
    method<-calculate_cosine
  else
    method<-calculate_euclidean
  
  # Calculates the distance matrix for each test data point to every training data point
  distance_matrix = matrix(nrow = nrow(x_test), ncol = nrow(x_train))
  for(i in 1:nrow(x_test)){
    for(j in 1:nrow(x_train)){
      distance_matrix[i,j] <- method(x_test[i,], x_train[j,])
    }
  }
  # sorted indices for all testing data points
  sorted_matrix<-t(apply(distance_matrix,1,order))
  # matrix of first k indices
  knn_index_matrix<-sorted_matrix[,1:k]
  # predicted class for each test data point based on majority
  predicted_classes<-t(apply(knn_index_matrix,1,get_class, y_train))
  return(as.factor(predicted_classes))
}

# ------- Part B -------

dtree <- function(x_train, y_train, x_test){
  
  # merging dataset with actual class values 
  input = cbind(x_train,actual_class=y_train)
  # building decision tree using formula as actual class values and split as information
  decision_tree = rpart(actual_class~. , data = input, method = "class", parms = list(split = "information"))
  # predicted classes using dtree
  predicted_classes = predict(decision_tree,x_test, type="class")
  return(predicted_classes)
}
# ------- Part C -------

generate_k_folds <- function(n, k) {
  # generates a sample of n values from a repeated sequence of 1 to k values
  fold_assignment = c(sample(rep(1:k, n/k)))
  return(fold_assignment)
}

k_fold_cross_validation_prediction <- function(x, y, k, k_folds, classifier_function) {
 
  total_predicted = array()
  # for every fold we generate the training and testing datasets using the generated random k fold
  for(j in 1:k)
  {
    test_indices = which(k_folds %in% j)
    new_x_test = x[test_indices,]
    new_x_train = x[-test_indices,]
    new_y_train = y[-test_indices]
    # we predict the class using the provided classifier function i.e knn or dtree 
    predicted_class = classifier_function(new_x_train,new_y_train,new_x_test)
    # concatenating all the predicted classes for each fold 
    predicted_class = array(predicted_class)
    # assigning the predicted classes to the right position in the resultant array
    total_predicted[test_indices] = predicted_class
  }
  return(total_predicted)
}

# ------- Part D -------

calculate_confusion_matrix <- function(y_pred, y_true){
 
  confusion_matrix = confusionMatrix(factor(y_pred),factor(y_true))
  return(confusion_matrix$table)
}

calculate_accuracy <- function(confusion_matrix){
 # accuracy = tp + tn / tp + tn + fp + fn
  accuracy = sum(diag(confusion_matrix))/sum(confusion_matrix)
  return(accuracy)
}


calculate_recall <- function(confusion_matrix){
  # recall = tp / tp + fn
  recall = confusion_matrix[2,2]/sum(confusion_matrix[,2])
  return(recall)
}

calculate_precision <- function(confusion_matrix){
  # precision = tp / tp + fp 
  precision = confusion_matrix[2,2]/sum(confusion_matrix[2,])
  return(precision)
}
