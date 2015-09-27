---
title: "Practical Machine Learning - Course Project"
author: "Jordi Guillaumes Pons"
date: "22 de setembre de 2015"
output: html_document
---

# Abstract

According to the course project requirements, this document details the steps made to find a suitable predictive model to the physical activity dataset provided and to predict the corresponding outcome for the testing dataset. 

The required R packages and libraries are as follow:


```r
require(plyr)
require(dplyr)
require(caret)
require(ggplot2)
require(rpart)
require(ipred)
require(randomForest)
require(survival)
require(gbm)
require(MASS)
require(klaR)
require(knitr)
```

# Exploratory analysis and preprocessing

The training dataset contains 19,622 observations of 160 variables. Most of those variables do not contain information for most of the observations. Our first processing of the training data consists on keeping the variables which are informed for all the observations. That leaves us with 29 variables, excluding the one which contains the outcome. The "testing" data set also provided does not contain the outcome, so it can not be used to test/validate our models. We will split the so called "training" data set into our own training and test data.


```r
# Cleaning data - this function selects just the variables with
# data available for all the observations, EXCLUDING the outcome
# values (classe)
cleanData <- function(df) {
  cd <- dplyr::select(df,
           roll_belt,pitch_belt, yaw_belt, total_accel_belt,
           gyros_belt_x, gyros_belt_y, gyros_belt_z,
           accel_belt_x, accel_belt_y, accel_belt_z,
           magnet_belt_x, magnet_belt_y, magnet_belt_z,
           roll_arm, pitch_arm, yaw_arm, total_accel_arm, 
           gyros_arm_x, gyros_arm_y, gyros_arm_z,
           accel_arm_x, accel_arm_y, accel_arm_z,
           magnet_arm_x, magnet_arm_y, magnet_arm_z,
           roll_dumbbell, pitch_dumbbell, yaw_dumbbell)
  cd
}
fulltraining <- read.csv("data/pml-training.csv")
set.seed(1234)
inTrain <- createDataPartition(y=fulltraining$classe,p=0.6,list=FALSE)
clean <- cleanData(fulltraining)
ctraining <- clean[inTrain,]
ctesting <- clean[-inTrain,]
outtrain <- fulltraining$classe[inTrain]
outtest  <- fulltraining$classe[-inTrain]
str(ctraining)
```

```
## 'data.frame':	11776 obs. of  29 variables:
##  $ roll_belt       : num  1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.45 1.45 1.43 ...
##  $ pitch_belt      : num  8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.17 8.18 8.18 ...
##  $ yaw_belt        : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt: int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x    : num  0.02 0 0.02 0.02 0.02 0.02 0.02 0.03 0.03 0.02 ...
##  $ gyros_belt_y    : num  0 0 0 0.02 0 0 0 0 0 0 ...
##  $ gyros_belt_z    : num  -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 0 -0.02 -0.02 ...
##  $ accel_belt_x    : int  -22 -20 -22 -21 -21 -22 -22 -21 -21 -22 ...
##  $ accel_belt_y    : int  4 5 3 2 4 3 4 4 2 2 ...
##  $ accel_belt_z    : int  22 23 21 24 21 21 21 22 23 23 ...
##  $ magnet_belt_x   : int  -7 -2 -6 -6 0 -4 -2 -3 -5 -2 ...
##  $ magnet_belt_y   : int  608 600 604 600 603 599 603 609 596 602 ...
##  $ magnet_belt_z   : int  -311 -305 -310 -302 -312 -311 -313 -308 -317 -319 ...
##  $ roll_arm        : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm       : num  22.5 22.5 22.1 22.1 22 21.9 21.8 21.6 21.5 21.5 ...
##  $ yaw_arm         : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x     : num  0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y     : num  -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 ...
##  $ gyros_arm_z     : num  -0.02 -0.02 0.02 0 0 0 0 -0.02 0 0 ...
##  $ accel_arm_x     : int  -290 -289 -289 -289 -289 -289 -289 -288 -290 -288 ...
##  $ accel_arm_y     : int  110 110 111 111 111 111 111 110 110 111 ...
##  $ accel_arm_z     : int  -125 -126 -123 -123 -122 -125 -124 -124 -123 -123 ...
##  $ magnet_arm_x    : int  -369 -368 -372 -374 -369 -373 -372 -376 -366 -363 ...
##  $ magnet_arm_y    : int  337 344 344 337 342 336 338 334 339 343 ...
##  $ magnet_arm_z    : int  513 513 512 506 513 509 510 516 509 520 ...
##  $ roll_dumbbell   : num  13.1 12.9 13.4 13.4 13.4 ...
##  $ pitch_dumbbell  : num  -70.6 -70.3 -70.4 -70.4 -70.8 ...
##  $ yaw_dumbbell    : num  -84.7 -85.1 -84.9 -84.9 -84.5 ...
```

The summary suggest that the ranges of the different variables are quite dissimilar, so it is a good idea to center and mean normalize them. We also have still 29 variables, which is a quite big number, so we will reduce the number of effective variables using PCA, selecting the necessary components to explain 95% of the variance.


```r
preProc <- preProcess(ctraining,method=c("center","scale","pca"),thresh=0.95)
preTraining <- predict(preProc, ctraining)
preProc
```

```
## 
## Call:
## preProcess.default(x = ctraining, method = c("center", "scale",
##  "pca"), thresh = 0.95)
## 
## Created from 11776 samples and 29 variables
## Pre-processing: centered, scaled, principal component signal extraction 
## 
## PCA needed 14 components to capture 95 percent of the variance
```

So we have now 14 variables to consider, which is a good reduction over the original 29. 

# Model selection

The outcome variable we want to predict is a categorical (factor) variable, so the models which give us a continuous value, as Linear Regression, are excluded. We will try to fit models using the methods which are suitable to classification problems:

- Trees (rpart)
- Bagging (treebag)
- Random Forests (rf)
- Boostig with trees (gbm)
- Linear Discriminant Analysis (lda)
- Naive Bayesian Analysis (nb)∫

## Model computation

We will use the train function of the caret package, which takes care of doing the resampling and cross-validation necessary to adjust the models. The default values used by test are usually good enough for that task, so we will not override them. Please take note that running the train functions takes a while, specially for Random Forests and Boosting, so please be patient.


```r
modRpart    <- train(outtrain ~ .,data=preTraining, method="rpart")
```

```
## Loading required package: rpart
```

```r
modTreebag  <- train(outtrain ~ .,data=preTraining, method="treebag")
```

```
## Loading required package: ipred
## Loading required package: plyr
## -------------------------------------------------------------------------
## You have loaded plyr after dplyr - this is likely to cause problems.
## If you need functions from both plyr and dplyr, please load plyr first, then dplyr:
## library(plyr); library(dplyr)
## -------------------------------------------------------------------------
## 
## Attaching package: 'plyr'
## 
## The following objects are masked from 'package:dplyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
```

```r
modRForest  <- train(outtrain ~ .,data=preTraining, method="rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
modGbmBoost <- train(outtrain ~ .,data=preTraining, method="gbm", verbose=FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
```

```r
modLda      <- train(outtrain ~ .,data=preTraining, method="lda")
```

```
## Loading required package: MASS
## 
## Attaching package: 'MASS'
## 
## The following object is masked from 'package:dplyr':
## 
##     select
```

```r
modNaive    <- train(outtrain ~ .,data=preTraining, method="nb")                     
```

```
## Loading required package: klaR
```

The accuracies for each of these models are the following:

| Method | Accuracy |
| ------ | -------: |
Trees | 0.3766387
Bagging | 0.8512459
Random Forests | 0.8939336
Boosting | 0.7199009
Linear Discriminant Analysis | 0.4451975
Naive Bayesian | 0.5392365

Perhaps unsurprisely the method which gives better accuracy is **Random Forests**:


```r
modRForest
```

```
## Random Forest 
## 
## 11776 samples
##    13 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.8939336  0.8659561  0.004406795  0.005566411
##    8    0.8874215  0.8577010  0.004485022  0.005647152
##   14    0.8631574  0.8270469  0.004915486  0.006197045
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

## Model validation

We will now use our test dataset (the one built by ourselves, not the one provided) to verify the model works reasonabily well.


```r
preTesting <- predict(preProc, newdata=ctesting)
preds      <- predict(modRForest, newdata=preTesting)
confusionMatrix(data = preds, reference = outtest)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2186   27   11   11    2
##          B   15 1453   23    7    9
##          C   15   28 1317   19   14
##          D   13    6   11 1243   13
##          E    3    4    6    6 1404
## 
## Overall Statistics
##                                          
##                Accuracy : 0.969          
##                  95% CI : (0.965, 0.9728)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : <2e-16         
##                                          
##                   Kappa : 0.9608         
##  Mcnemar's Test P-Value : 0.139          
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9794   0.9572   0.9627   0.9666   0.9736
## Specificity            0.9909   0.9915   0.9883   0.9934   0.9970
## Pos Pred Value         0.9772   0.9642   0.9454   0.9666   0.9866
## Neg Pred Value         0.9918   0.9897   0.9921   0.9934   0.9941
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2786   0.1852   0.1679   0.1584   0.1789
## Detection Prevalence   0.2851   0.1921   0.1775   0.1639   0.1814
## Balanced Accuracy      0.9852   0.9743   0.9755   0.9800   0.9853
```

As can be seen, we have a good accuracy with our test dataset, so we can conclude this model is good enough to predict the outcomes for the *real* test data.

# Application to the problem test data

We just need to generate the predicted outcomes for the problem test data to finish the assignement.


```r
fullptest <- read.csv("data/pml-testing.csv")
cptest <- cleanData(fullptest)
preCptest <- predict(preProc, newdata=cptest)
answers   <- predict(modRForest, newdata=preCptest)
table(answers)
```

```
## answers
## A B C D E 
## 7 7 2 1 3
```
