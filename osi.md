Online Shoppers Intention classification using Boosting
================
Nicola Disabato
2022-08-01

## Importing libraries

``` r
library(magrittr) 
library(dplyr)
```

    ## 
    ## Caricamento pacchetto: 'dplyr'

    ## I seguenti oggetti sono mascherati da 'package:stats':
    ## 
    ##     filter, lag

    ## I seguenti oggetti sono mascherati da 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(caret)
```

    ## Caricamento del pacchetto richiesto: ggplot2

    ## Caricamento del pacchetto richiesto: lattice

``` r
library(fastAdaboost)
library(Matrix)
library(ROCR)
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Caricamento pacchetto: 'pROC'

    ## I seguenti oggetti sono mascherati da 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(xgboost)
```

    ## 
    ## Caricamento pacchetto: 'xgboost'

    ## Il seguente oggetto è mascherato da 'package:dplyr':
    ## 
    ##     slice

``` r
library(gbm)
```

    ## Loaded gbm 2.1.8

## Importing the dataset

``` r
# Load the dataset and explore
intentions <- read.csv("online_shoppers_intention.csv", header = TRUE) 
str(intentions)
```

    ## 'data.frame':    12330 obs. of  18 variables:
    ##  $ Administrative         : int  0 0 0 0 0 0 0 1 0 0 ...
    ##  $ Administrative_Duration: num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Informational          : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Informational_Duration : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ ProductRelated         : int  1 2 1 2 10 19 1 0 2 3 ...
    ##  $ ProductRelated_Duration: num  0 64 0 2.67 627.5 ...
    ##  $ BounceRates            : num  0.2 0 0.2 0.05 0.02 ...
    ##  $ ExitRates              : num  0.2 0.1 0.2 0.14 0.05 ...
    ##  $ PageValues             : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ SpecialDay             : num  0 0 0 0 0 0 0.4 0 0.8 0.4 ...
    ##  $ Month                  : chr  "Feb" "Feb" "Feb" "Feb" ...
    ##  $ OperatingSystems       : int  1 2 4 3 3 2 2 1 2 2 ...
    ##  $ Browser                : int  1 2 1 2 3 2 4 2 2 4 ...
    ##  $ Region                 : int  1 1 9 2 1 1 3 1 2 1 ...
    ##  $ TrafficType            : int  1 2 3 4 4 3 3 5 3 2 ...
    ##  $ VisitorType            : chr  "Returning_Visitor" "Returning_Visitor" "Returning_Visitor" "Returning_Visitor" ...
    ##  $ Weekend                : logi  FALSE FALSE FALSE FALSE TRUE FALSE ...
    ##  $ Revenue                : logi  FALSE FALSE FALSE FALSE FALSE FALSE ...

``` r
table(intentions$Revenue)
```

    ## 
    ## FALSE  TRUE 
    ## 10422  1908

As you can see, the dataset is made up of 12330 instances and 18
features. Of these observations, only 1908 are users who have finalized
a purchase.

All details can be found through the following link:
<https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#>

## Data Preparation

``` r
intentions <- intentions %>% 
  mutate(OperatingSystems = as.factor(OperatingSystems),
         Browser = as.factor(Browser),
         Region = as.factor(Region),
         TrafficType = as.factor(TrafficType),
         VisitorType = as.factor(VisitorType),
         Month = as.factor(Month)
         )

intentions <- intentions %>% 
  mutate(
         Weekend = as.numeric(Weekend),
         Revenue = as.numeric(Revenue)
         )

str(intentions)
```

    ## 'data.frame':    12330 obs. of  18 variables:
    ##  $ Administrative         : int  0 0 0 0 0 0 0 1 0 0 ...
    ##  $ Administrative_Duration: num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Informational          : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Informational_Duration : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ ProductRelated         : int  1 2 1 2 10 19 1 0 2 3 ...
    ##  $ ProductRelated_Duration: num  0 64 0 2.67 627.5 ...
    ##  $ BounceRates            : num  0.2 0 0.2 0.05 0.02 ...
    ##  $ ExitRates              : num  0.2 0.1 0.2 0.14 0.05 ...
    ##  $ PageValues             : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ SpecialDay             : num  0 0 0 0 0 0 0.4 0 0.8 0.4 ...
    ##  $ Month                  : Factor w/ 10 levels "Aug","Dec","Feb",..: 3 3 3 3 3 3 3 3 3 3 ...
    ##  $ OperatingSystems       : Factor w/ 8 levels "1","2","3","4",..: 1 2 4 3 3 2 2 1 2 2 ...
    ##  $ Browser                : Factor w/ 13 levels "1","2","3","4",..: 1 2 1 2 3 2 4 2 2 4 ...
    ##  $ Region                 : Factor w/ 9 levels "1","2","3","4",..: 1 1 9 2 1 1 3 1 2 1 ...
    ##  $ TrafficType            : Factor w/ 20 levels "1","2","3","4",..: 1 2 3 4 4 3 3 5 3 2 ...
    ##  $ VisitorType            : Factor w/ 3 levels "New_Visitor",..: 3 3 3 3 3 3 3 3 3 3 ...
    ##  $ Weekend                : num  0 0 0 0 1 0 0 1 0 0 ...
    ##  $ Revenue                : num  0 0 0 0 0 0 0 0 0 0 ...

We use one-hot encoding for categorical variables.

``` r
dmy <- dummyVars(" ~ .", data = intentions)
intentions <- data.frame(predict(dmy, newdata = intentions))

dim(intentions)
```

    ## [1] 12330    75

After the preprocessing phase, we obtain a dataset with 75 features.

## Train and Test partition

``` r
set.seed(100)
inTrain <- createDataPartition(y = intentions$Revenue, p = .75, list = FALSE)
train <- intentions[ inTrain,] 
test <- intentions[-inTrain,]

X_train <- sparse.model.matrix(Revenue ~ .-1, data = train)
y_train <- train[,"Revenue"]  
X_test <- sparse.model.matrix(Revenue~.-1, data = test)
y_test <- test[,"Revenue"]
```

## Let’s explore AdaBoost model

``` r
model_adaboost <- adaboost(Revenue ~ ., data=train, nIter=10)
model_adaboost
```

    ## adaboost(formula = Revenue ~ ., data = train, nIter = 10)
    ## Revenue ~ .
    ## Dependent Variable: Revenue
    ## No of trees:10
    ## The weights of the trees are:1.3226071.1136721.0410931.0339541.0131450.95893290.92418360.92972750.90278420.8971979

After training the model on the train dataset, you can use the predict
() method to predict the output of the Revenue class in the test
dataset. To analyze the performance of the model, it was decided to
print the confusion matrix in addition to the precision, recall and
f1-score metrics, in addition to the accuracy metric.

``` r
#predictions
pred_ada = predict(model_adaboost, newdata=test)

#confusion matrix creation
cm = confusionMatrix(as.factor(pred_ada$class),as.factor(y_test), positive = '1')
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 2431  204
    ##          1  141  306
    ##                                          
    ##                Accuracy : 0.8881         
    ##                  95% CI : (0.8764, 0.899)
    ##     No Information Rate : 0.8345         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.5736         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.0008439      
    ##                                          
    ##             Sensitivity : 0.60000        
    ##             Specificity : 0.94518        
    ##          Pos Pred Value : 0.68456        
    ##          Neg Pred Value : 0.92258        
    ##              Prevalence : 0.16548        
    ##          Detection Rate : 0.09929        
    ##    Detection Prevalence : 0.14504        
    ##       Balanced Accuracy : 0.77259        
    ##                                          
    ##        'Positive' Class : 1              
    ## 

``` r
print(cm$byClass[5])
```

    ## Precision 
    ## 0.6845638

``` r
print(cm$byClass[6])
```

    ## Recall 
    ##    0.6

``` r
print(cm$byClass[7])
```

    ##        F1 
    ## 0.6394984

It is immediately evident how the classification model, despite having
an accuracy value of 0.88, is found to be not very precise as the values
of Precision, Recall and F1 are quite low. This often happens in these
cases, that is with the presence of unbalanced classes: in such cases
the accuracy metric is not very informative.

To build the best possible Adaboost model, depending on the dataset, we
have chosen to build a graph from which to display the best number of
decision trees (of iterations) to specify to build a more precise model,
based on the errors made.

``` r
best_adaboost <- adaboost(Revenue ~ ., data=train, nIter=125)

#predictions
pred_best_ada = predict(best_adaboost, newdata=test)

#confusion matrix creation
cm <- confusionMatrix(as.factor(pred_best_ada$class),as.factor(y_test), positive = '1')
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 2474  209
    ##          1   98  301
    ##                                           
    ##                Accuracy : 0.9004          
    ##                  95% CI : (0.8893, 0.9107)
    ##     No Information Rate : 0.8345          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6049          
    ##                                           
    ##  Mcnemar's Test P-Value : 3.429e-10       
    ##                                           
    ##             Sensitivity : 0.59020         
    ##             Specificity : 0.96190         
    ##          Pos Pred Value : 0.75439         
    ##          Neg Pred Value : 0.92210         
    ##              Prevalence : 0.16548         
    ##          Detection Rate : 0.09766         
    ##    Detection Prevalence : 0.12946         
    ##       Balanced Accuracy : 0.77605         
    ##                                           
    ##        'Positive' Class : 1               
    ## 

``` r
print(cm$byClass[5])
```

    ## Precision 
    ##  0.754386

``` r
print(cm$byClass[6])
```

    ##    Recall 
    ## 0.5901961

``` r
print(cm$byClass[7])
```

    ##        F1 
    ## 0.6622662

From the results it is possible to observe how better results have been
achieved, starting from the Precision metric which describes a greater
precision in the prediction of the positive label 1 which passes from
0.68 to 0.75. The other metrics remain similar.
