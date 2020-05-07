
source('./hw2_solution.R')

# read data from disk, extract train test into separate variables 
all_data <- read.csv('./pima-indians-diabetes.csv', stringsAsFactors= T, header = F)
x <- all_data[, 1:8]
y <- as.factor(all_data[, 9])

# ------- Part A -------

# Train a 5-nn classifier on the whole dataset and predict on the training dataset
train_knn_predict <- knn(x, y, x, k=3)
# Calcualte training accuracy. Expected output: 0.77
mean(train_knn_predict == y)

# ------- Part B -------

# Train RPART on the whole dataset and predict on the training dataset
train_dtree_predict <- dtree(x, y, x)
# Calcualte training accuracy. Expected output: 0.85
mean(train_dtree_predict == y)

# ------- Part C -------

# Generate 5 folds for cross-validation
# We store the folds so we can use the same ones for both classifiers
set.seed(123) # The folds are random, so set a seed for consistency
k <- 5
folds <- generate_k_folds(nrow(x), k)
# There should be 20 of each number 1 through 5
table(folds)

# calculate cross validation results for decision 3-nn
dt_cv_result <- k_fold_cross_validation_prediction(x, y, k, folds, dtree)
# calculate cross validation results for decision tree
knn_cv_result <- k_fold_cross_validation_prediction(x, y, k, folds, knn)

# Just for testing, we use simple (non-random) folds, which allows us to 
# confirm that we get the expected output from knn and dtree
simple_folds <- rep(1:k, 100/k)
# Check the percentage of Class 1 predictions
# Expected result: 0.3
mean(k_fold_cross_validation_prediction(x, y, k, simple_folds, dtree) == "1")

# Check the percentage of Class 1 predictions
# Expected result: 0.31
mean(k_fold_cross_validation_prediction(x, y, k, simple_folds, knn) == "1")


# ------- Part D -------

# evaluating the classification outcomes using accuracy, recall and precision
# note: exact values will depend on your the k folds you used
knn_cv_matrix <- calculate_confusion_matrix(knn_cv_result,y)
knn_cv_accuracy <- calculate_accuracy(knn_cv_matrix)
knn_cv_recall <- calculate_recall(knn_cv_matrix)
knn_cv_precision <- calculate_precision(knn_cv_matrix)

dt_cv_matrix <- calculate_confusion_matrix(dt_cv_result,y)
dt_cv_accuracy <- calculate_accuracy(dt_cv_matrix)
dt_cv_recall <- calculate_recall(dt_cv_matrix)
dt_cv_precision <- calculate_precision(dt_cv_matrix)
