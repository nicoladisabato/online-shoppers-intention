# Online Shoppers Intention

## Binary Classification using Boosting algorithms

## Goal

The purpose of this real business case is to build a predictive model that is able, given the characteristics of a user session, to predict if that user will finalize a purchase.

Each session is characterized by 18 variables, 10 of which are numerical and 8 are categorical: some of the aspects present are the number of pages visited in a category, the time spent on them, the month, the operating system, the browser and other.

The class variable is represented by Revenue, which indicates with 0 that the user has not finalized a purchase while with 1 the conclusion of the same.

We will use the Boosting technique that will improve the performance of a simple classifier, starting from weak learners. We will use the main boosting algorithms, such as XGBoost and AdaBoost:M1.
