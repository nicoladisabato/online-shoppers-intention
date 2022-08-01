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
