# BTMA 531 Assignment 2
# Cooper Chung (Redacted Student ID)
# cooper.chung1@ucalgary.ca

# Question 1a
shopper_data <- read.csv("online_shoppers_intention2.csv")          # Read the csv

set.seed(99)                                                        # Set seed for replicability

training_index <- sample(nrow(shopper_data), 10000, replace = F)    # Get 10000 random indices, with index values up to a max of 12330

training_data <- shopper_data[training_index,]                      # Generate subset of data from csv with random indices

fit1 <- glm(Revenue ~ ., family = binomial, data = training_data)   # Generate Logistic Regression

summary(fit1)

exp(fit1$coefficients["VisitorTypeReturning_Visitor"])/(1 + exp(fit1$coefficients["VisitorTypeReturning_Visitor"]))
# Calculating odds ratio for Returning Visitors result is 0.4214032. Since this is below 1, the odds are unlikely that Returning visitors will buy again.
# Compared to new visitors, they are 1 - 0.4214032 = 0.57... or 57% less likely to buy compared to new visitors.


# Question 1b
test_data <- shopper_data[-training_index,]                         # Create test dataset by subtracting our training indices

pred_test <- predict(fit1, test_data, type = "response")            # Predict our test dataset using the model generated earlier

pred_test <- ifelse(pred_test > 0.5, T, F)                          # Bayes Boundary


# Question 1c
mean(pred_test == test_data$Revenue)  # Gives us a prediction accuracy of 0.8927039

table(pred_test, test_data$Revenue)   # Confusion Matrix                         

1947 / (1947 + 47)  # Calculate Specificity -> True Negative / (True Negative + False Positive) = 0.9764293

133 / (133 + 203)   # Calculate Sensitivity -> True Positive / (True Positive + False Negative) = 0.3958333


# Question 1d
# Based off the dataset provided, evaluating the performance of the classifier is difficult when all you're looking at is Sensitivity, Specificity, and accuracy.
# The dataset also doesn't mention any mis-classification costs (learned from OPMA 419), meaning avoiding-false positives is essentially
# negligible - ruling out the need to improve specificity. In terms of sensitivity, the performance here is good at a 0.97. Identifying true positives can
# help us determine what results in a Revenue = TRUE, and expand on that aspect. In that sense, the performance of our classifier is actually quite good.
# One thing that I think we can include is the use of a threshold (learned from OPMA 419), or lowering the threshold we were originally using.
# This can help us expand what we classify as a Revenue = TRUE, since there is no mentioned mis-classification cost. The following is what I 
# would do to implement this change. I would lower our gate from 0.5 to 0.4.
pred_test <- predict(fit1, test_data, type = "response")  # Re-do this step to prep for lower boundary

pred_test <- ifelse(pred_test > 0.4, T, F)  # Lower boundary from 0.5 to 0.4

mean(pred_test == test_data$Revenue)        # Recalculated Accuracy = 0.8935622. We can see that the performance has just marginally improved

table(pred_test, test_data$Revenue)         # Confusion Matrix. From this, we can see that our Type 1 Error is 62, and our Type 2 Error is 186


# Question 2a
library(MASS)                                                         # Load MASS library to use LDA function

accent_data <- read.csv("accent-mfcc-data-1.csv")                     # Read data into R

set.seed(99)                                                          # Set seed for replicability

accent_training_index <- sample(nrow(accent_data), 250, replace = F)  # Get 250 random indices, with index values up to a max of 329

accent_training_data <- accent_data[accent_training_index,]           # Generate subset of data from csv with random indices

fit2 <- lda(language ~ ., data = accent_training_data)                # Create LDA classifier


# Question 2b
accent_test_data <- accent_data[-accent_training_index,]                # Create test dataset by subtracting our training indices

accent_pred_test <- predict(fit2, accent_test_data, type = "response")  # Predict test dataset using LDA model

accent_pred_test$class  # Class predictions via LDA


# Question 2c
mean(accent_pred_test$class == accent_test_data$language) # Gives us an accuracy of 0.6582278

table(accent_pred_test$class, accent_test_data$language)  # Creates confusion matrix


# Question 2d
fit3 <- qda(language ~ ., data = accent_training_data)    # Create QDA classifier


# Question 2e
qda_accent_pred_test <- predict(fit3, accent_test_data, type = "response")  # Predict test dataset using QDA model

qda_accent_pred_test$class  # Class predictions via QDA


# Question 2f
mean(qda_accent_pred_test$class == accent_test_data$language) # Gives us an accuracy of 0.6835443
# From our results, we can see that the QDA model performs better than the LDA model in terms of accuracy, but not by too much, as it is around 3% better.
# What this may suggest about the dataset is in the name of the analysis, QDA or QUADRATIC discriminant analysis performs better meaning that
# the dataset may not be linear in terms of the decision boundary. A linear combination of the classes may not be sufficient to create a decision boundary.
# QDA performing better may also suggest that each class has its own covariance matrix. Some other steps we could take to create a more robust model is to 
# generate some visualizations, compare more than just a LDA and QDA model, incorporate some more metrics to keep track of, and incorporate some
# cross-validations (OPMA 419)


# Question 2g
library(class)                                    # Load class library to use knn function

knn_train <- accent_training_data[,-1]            # Prep training data without target column

knn_test <- accent_test_data[,-1]                 # Prep test data without target column

knn_train_scaled <- scale(knn_train)              # Scale training data

means <- attr(knn_train_scaled, "scaled:center")  # Create list of means

sds <- attr(knn_train_scaled, "scaled:scale")     # Create list of standard deviations

knn_test_scaled <- scale(knn_test, center = means, scale = sds) # Scale the test dataset with the scaling model we get from the training dataset

knn_target <- accent_training_data$language  # Designate the language column from unscaled training data as target

fit4 <- knn(train = knn_train_scaled, test = knn_test_scaled, cl = knn_target, k = 5)   # Create KNN model with k = 5

mean(fit4 == accent_test_data$language) # Gives us an accuracy of 0.6962025

fit5 <- knn(train = knn_train_scaled, test = knn_test_scaled, cl = knn_target, k = 10)  # Create KNN model with k = 10

mean(fit5 == accent_test_data$language) # Gives us an accuracy of 0.6835443

# From these two models, we can see that the model with k = 5 performed better with a higher accuracy. The lower K value performing better may be
# a result of there being a less defined boundary, allowing the better encapsulation of each data point.


# Question 3a
library(tree) # Load tree library to use tree function

car_data <- read.csv("CarEvals.csv", header = T, colClasses = c("factor", "factor", "factor",
                                                                "factor", "factor", "factor",
                                                                "factor"))  # Read data into R

tree_car <- tree(Class ~ ., data = car_data)  # Create Classification Tree

plot(tree_car)  # Plot created tree

text(tree_car, pretty = 0, cex = 0.7)  # Add text to plot


# Question 3b
set.seed(99)                                                # Set seed for replicability

car_index <- sample(nrow(car_data), 1000, replace = F)      # Get 1000 random indices, with index values up to a max of 1719

car_training_data <- car_data[car_index,]                   # Generate subset of data from csv with random indices

tree_car_1000 <- tree(Class ~ ., data = car_training_data)  # Generate tree with training data

car_test_data <- car_data[-car_index,]                      # Create test dataset by subtracting our training indices

car_pred <- predict(tree_car_1000, car_test_data, type = "class")  # Predict test dataset using tree model


# Question 3c
mean(car_pred == car_test_data$Class) # Gives us an accuracy of 0.9082058

table(car_pred, car_test_data$Class)  # Creates confusion matrix


# Question 3d
set.seed(99)  # Set seed for replicability

tree_car_1000_cv <- cv.tree(tree_car_1000, FUN = prune.misclass)  # Use cross-validation to prune the tree

plot(tree_car_1000_cv$size, tree_car_1000_cv$dev, type = "b", xlab = "Tree Size", ylab = "Cross-Validation Error",
     main = "Cross-Validation Error and Tuning Parameter vs Tree Size") # Create plot

tree_car_1000_cv$size[which.min(tree_car_1000_cv$dev)]  # Find the best tree size that minimizes the error - 16


# Question 3e
tree_car_1000_16 <- prune.tree(tree_car_1000, best = 16)  # Prune tree specifying the size

car_pred_16 <- predict(tree_car_1000_16, car_test_data, type = "class")  # Predict test dataset using this new tree model

mean(car_pred_16 == car_test_data$Class)  # Gives us an accuracy of 0.9082058, unchanging from before


# Question 4a
# If I wanted to build a classification model to predict y, I would use a logistic regression. Logistic regressions are appropriately used for predicting
# a categorical (in this case - binary) output variables. While LDA could be considered since it is logistic but better/more stable, it shouldn't be used because
# we only have 2 levels we are trying to predict, not more than 2. We also don't know if the classes are well-separated, and if the predictors are approximately normal.
# Using a logistic regression will also provide us with the probabilities of belonging to either 1 or 0 for y, which helps us understand which predictors
# should be included in the model, and which ones don't matter at all/as much.


# Question 4b
# In the context of this dataset, type 1 errors (False Positives) occur when the model classifies an individual "yes" as a subscriber,
# but in reality that individual is not. Type 2 errors (False Negatives) occur when the model classifies an individual "no" as a subscriber, but in reality they are.
# Costs for both of these errors would most likely be financial in nature, regarding opportunity and sunk costs. For type 1 errors, the bank may expend resources
# (money, marketing, employee hours, etc...) in marketing the term deposit to an individual since they were marked to be "yes" as a subscriber, only for the
# individual to not subscribe. In a sense, this would waste those resources, time, and money, making this the cost for type 1 errors. For type 2 errors, this is solely
# opportunity cost. The bank would miss an opportunity for the individual to subscribe, as the individual was marked as a "no" from the model, but in reality they would
# have subscribed. In a sense, the bank misses out on any potential revenues from this individual.

# Deciding on which one is more costly depends on the bank, and what financial figures they would like to tie to either error. It depends on how much they would spend
# on marketing, and how much revenue they would gain if a customer did in fact subscribe. They would also have to weigh the costs of either error, and see which error
# would be more important for them to minimize. If gaining a customer and generating revenue from them is more profitable in the long or short term,
# and the cost of gaining a customer is much lower than the potential revenues, I would argue that a type 2 error would be more costly. If the bank is expending a
# considerable amount of resources to gain one customer, and the revenues are not that high, I would argue that a type 1 error would be more costly. In any case
# I will consider a type 2 error to be more costly.

# To reduce type 2 errors, I would seek to find an optimal threshold to classify as either a yes or no (learned from OPMA 419). I would also pair this optimization
# with focusing on the specificity of the dataset. Generally speaking, I would look for a specificity that isn't too high (>95%) and also not too low (<80%).
# Balancing specificity with revenues is the key here, as specificity will help reduce type 2 errors, and thus, reduce the opportunity cost the bank undertakes.


# Question 4c
# In creating a realistic predictive model for classification, there are some variables that are nonsensical to include, and some variables that may provide some
# insight, but shouldn't be used due to their societal implications. Some variables that can give us some information or warning signs, but shouldn't be used in
# the model are Age, Job, Marital Status, and Education. These indicators shouldn't be used because they can be dynamic and change quickly over time. These variables
# can also be correlated with each other, which is undesirable for predictors. These variables may also contain some outliers, especially in the age category where
# there may be occurrences that don't follow other occurrences in the same age group. This may also lead to some possible ethical and societal profiling implications
# that companies are better off just avoiding (customer complaints about racial, income, age profiling, etc...).

# The other two variables that simply shouldn't be used due to it being completely irrelevant is duration and voicemail. The duration of a call, and whether or not it
# went to voicemail has no real world indication of whether or not an individual will subscribe.
