---
title: 'Peer-graded Assignment: Prediction Assignment Writeup'
author: "Gaspare Mattarella"
date: "9/5/2020"
output: 
  html_document: 
    df_print: tibble
    fig_caption: yes
    keep_md: yes
    toc: yes
---



## Introduction
This report is prepared as one of the requirement in Practical Machine Learning online course by Johns Hopkins University. The basic goal of this assignment is to predict the manner of the subject (6 participants) performed some exercise. For this assignment, in order to predict the manner of the subject did the exercise decision tree and random forest method will be performed to determine the best prediction. The best prediction is determined by the highest accuracy.

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here

## Pre-processing
This segment includes packages required and uploading data into r.

Packages needed to perform tree classification, bagging, random forest are as follows:


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(readr)
library(RColorBrewer)
library(RGtk2)
library(rpart)
library(rpart.plot)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(gbm)
```

```
## Loaded gbm 2.1.5
```


## Data Cleaning

Many columns of the data set contain the same value accros the lines. These “near-zero variance predictors”" bring almost no information to our model and will make computing unnecessarily longer. Others are entirely filled with NA values. Finnaly, the six first variables do not concern fitness motions whatsoever. They also need to be remove before we start fitting our model.
Note that 93 features out of 160 contain missing values, and each of these features contains 9407 missing values out of 9619 observations, i.e. these features have approximately 97.8% of missing values. Given this abundance of missing values we have decided to not consider these features in our analysis.

```r
w <- sapply(training_pml,function(x) mean(is.na(x)) > 0.95) 
b <- sapply(testing_pml,function(x) mean(is.na(x)) > 0.95)

training_pml <- training_pml[,w == F]
testing_pml <- testing_pml[,w == F]
```

In order to keep consistency, we apply the exact same pre-processing to the testing


We will mostly focus on the training data set while building our models; we will use the testing set for cross-validation

```r
inTrain <- createDataPartition(y= training_pml$classe, p = .65, list =F)
train <- training_pml[inTrain,]
test <- training_pml[-inTrain,]
```
Furthermore, the first feature is simply the unique sequential number of the observation, which seems irrelevant for predicting the quality of exercise. The sixth feature new_window represents a new time window for sliding window feature extraction, and it is highly skewed with 9419 “no”s out of 9619 observations (i.e. 98% “no”s), so we have also decided to discard it from our models. 

```r
train <- train[-c(1:7)]
test <- test[-c(1:7)]
```
This concludes the feature selection of our data.

## Prediction model
Since we want to predict the class (A, B, C, D, E) to which an observation belongs, this is a classification problem with multiple classes. Therefore, there are several models we can use here. We have chosen the following:

1. Tree Classification.

2. Bagging.

3. Random forest model.

The first model we're goning to fit is a simple Tree Classification:


```r
tree <- train(factor(classe)~ ., method= "rpart", data = train,
              metric = "Accuracy", 
                              preProcess=c("center", "scale"),
                              trControl=trainControl(method = "cv"
                                                       , number = 4
                                                       , p= 0.60
                                                       , allowParallel = TRUE))
tree
```

```
## CART 
## 
## 12757 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (51), scaled (51) 
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 9569, 9567, 9568, 9567 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.02979189  0.5277950  0.39310109
##   0.03493976  0.5224659  0.38589808
##   0.06336254  0.3271828  0.07161085
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.02979189.
```

```r
fancyRpartPlot(tree$finalModel)
```

![](Assignment_files/figure-html/ftn-1.png)<!-- -->
As we can see the accuracy is the model is just 51% which makes evident this is not a proper model to fit this kind of data.

The next models we're going to fit are the "bagging" and the random forest, which, according to my previsons will be way much better models in evaluate our model.


```r
baggi <- train(factor(classe)~ ., method= "treebag", data = train,
              metric = "Accuracy", 
                              preProcess=c("center", "scale"),
                              trControl=trainControl(method = "cv"
                                                       , number = 4
                                                       , p= 0.60
                                                       , allowParallel = TRUE))
baggi
```

```
## Bagged CART 
## 
## 12757 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (51), scaled (51) 
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 9568, 9568, 9568, 9567 
## Resampling results:
## 
##   Accuracy   Kappa   
##   0.9810304  0.976003
```

```r
bag_pred <- predict(baggi, test)
confusionMatrix(bag_pred,factor(test$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1947   13    0    2    2
##          B    6 1298   12    0    1
##          C    0   15 1178    5    2
##          D    0    1    7 1115    3
##          E    0    1    0    3 1254
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9894          
##                  95% CI : (0.9866, 0.9917)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9865          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9969   0.9774   0.9841   0.9911   0.9937
## Specificity            0.9965   0.9966   0.9961   0.9981   0.9993
## Pos Pred Value         0.9913   0.9856   0.9817   0.9902   0.9968
## Neg Pred Value         0.9988   0.9946   0.9966   0.9983   0.9986
## Prevalence             0.2845   0.1934   0.1744   0.1639   0.1838
## Detection Rate         0.2836   0.1891   0.1716   0.1624   0.1827
## Detection Prevalence   0.2861   0.1918   0.1748   0.1640   0.1832
## Balanced Accuracy      0.9967   0.9870   0.9901   0.9946   0.9965
```



```r
Random_Forest <- randomForest(factor(classe) ~ .,
                              data = train,
                              ntree=100
                              , metric = "Accuracy" 
                              , preProcess=c("center", "scale")
                              , trControl=trainControl(method = "cv"
                                                       , number = 4
                                                       , p= 0.60
                                                       , allowParallel = TRUE))
```

Both the models show much higher degrees of accuracy in the cross validation tests, both close to $100\%$. 

```r
plot(Random_Forest)
```

![](Assignment_files/figure-html/d-1.png)<!-- -->

```r
rf_pred <- predict(Random_Forest, test)
confusionMatrix(rf_pred,factor(test$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1952    2    0    0    0
##          B    1 1321    6    0    0
##          C    0    5 1189   10    0
##          D    0    0    2 1114    0
##          E    0    0    0    1 1262
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9961          
##                  95% CI : (0.9943, 0.9974)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.995           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9995   0.9947   0.9933   0.9902   1.0000
## Specificity            0.9996   0.9987   0.9974   0.9997   0.9998
## Pos Pred Value         0.9990   0.9947   0.9875   0.9982   0.9992
## Neg Pred Value         0.9998   0.9987   0.9986   0.9981   1.0000
## Prevalence             0.2845   0.1934   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1924   0.1732   0.1623   0.1838
## Detection Prevalence   0.2846   0.1934   0.1754   0.1626   0.1840
## Balanced Accuracy      0.9995   0.9967   0.9953   0.9949   0.9999
```
## Testing the model

Our out-of-sample accuracy is 0.9952, so our out-of-sample error is 0.0048. Moreover, both specificity and sensitivity are quite high for all classes, therefore our model seems to be performing very well in new data.


