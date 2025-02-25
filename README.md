# Berkeley Haas - Professional Certificate in Machine Learning and Artificial Intelligence - Cap

## Background
This project aims to train a classification model to predict if a user will accept a coupon given his/her answers to some survey questions.

The data used in here comes from a survey on Amazon Mechanical Turk. 
[In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)

In this project the work I have done is split into 3 parts.

### Part 1 - Preprocessing of the data
During this data processing, I have worked to:

* Handle Missing Values
* Handle Duplicated Data
* Handle Outliers
* Feature Transformation
* Feature Encoding
* Handle Class Imbalance
* Feature Selection & Feature Extraction

### Part 2 Modeling training

Here in Part 2, I have transformed the features  we tested our data train to machine learning model and evaluated it. 

#### Data Tranformation and Encoding
Here I am performing data preprocessing for machine learning pipeline using ColumnTransformer from sklearn. 

Standardized numerical features by scaling them to have zero mean and unit variance using StandardScaler().

Encoded categorical variables using OneHotEncoding, converting each category into a binary vector.

Fit the transformations to X_train (learns scaling parameters for numerical data and identifies unique categories for OneHotEncoding).

Used the same transformations learned from X_train and applies them to X_test (without refitting).

All numerical features are standardized.
All categorical features are one-hot encoded.
With this the data is now ready for machine learning models that require numerical input.

Applying StandardScaler and OneHotEncoding created a combination of Input features to form a complex set of categorical features. 

### Numerical Features
    ['temperature', 'age', 'education', 'income', 'Bar', 'CoffeeHouse',
       'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50', 'distance']

### Categorical Features

    'destination_Home' 'destination_No Urgent Place' 'destination_Work'
    'passanger_Alone' 'passanger_Friend(s)' 'passanger_Kid(s)'
    'passanger_Partner' 'weather_Rainy' 'weather_Snowy' 'weather_Sunny'
    'time_10AM' 'time_10PM' 'time_2PM' 'time_6PM' 'time_7AM' 'coupon_Bar'
    'coupon_Carry out & Take away' 'coupon_Coffee House'
    'coupon_Restaurant(20-50)' 'coupon_Restaurant(<20)' 'expiration_1d'
    'expiration_2h' 'gender_Female' 'gender_Male' 'maritalStatus_Divorced'
    'maritalStatus_Married partner' 'maritalStatus_Single'
    'maritalStatus_Unmarried partner' 'maritalStatus_Widowed'
    'has_children_0' 'has_children_1' 'occupation_Architecture & Engineering'
    'occupation_Arts Design Entertainment Sports & Media'
    'occupation_Building & Grounds Cleaning & Maintenance'
    'occupation_Business & Financial'
    'occupation_Community & Social Services'
    'occupation_Computer & Mathematical'
    'occupation_Construction & Extraction'
    'occupation_Education&Training&Library'
    'occupation_Farming Fishing & Forestry'
    'occupation_Food Preparation & Serving Related'
    'occupation_Healthcare Practitioners & Technical'
    'occupation_Healthcare Support'
    'occupation_Installation Maintenance & Repair' 'occupation_Legal'
    'occupation_Life Physical Social Science' 'occupation_Management'
    'occupation_Office & Administrative Support'
    'occupation_Personal Care & Service' 'occupation_Production Occupations'
    'occupation_Protective Service' 'occupation_Retired'
    'occupation_Sales & Related' 'occupation_Student'
    'occupation_Transportation & Material Moving' 'occupation_Unemployed'
    'direction_same_0' 'direction_same_1']

The total number of features after applying Scaling and OneHotEncodiing stands as 68. 

#### Creation and running performance of different models

 I created and tested the model performance for the following models:
 
 * Logistic Regression
 * Lasso Regression
 * Decision Trees
 * Gaussian Naive Bayes
 * Bernoulli Naive Bayes
 * K-nearest Neighbor
 * Linear SVM - Support Vector Machines

### Part 3 Applying PCA - Principal Component Analysis to reduce components / input features

Finally I explored PCA - Principal Component Analysis, to reduce the dimensions or in other words the number of features. 

Applying PCA reduced the features from 68 to 30 most influential ones listed below:
    PC1 is most influenced by: RestaurantLessThan20
    PC2 is most influenced by: age
    PC3 is most influenced by: temperature
    PC4 is most influenced by: income
    PC5 is most influenced by: distance
    PC6 is most influenced by: CoffeeHouse
    PC7 is most influenced by: Restaurant20To50
    PC8 is most influenced by: Restaurant20To50
    PC9 is most influenced by: Bar
    PC10 is most influenced by: education
    PC11 is most influenced by: expiration_2h
    PC12 is most influenced by: RestaurantLessThan20
    PC13 is most influenced by: RestaurantLessThan20
    PC14 is most influenced by: destination_Home
    PC15 is most influenced by: has_children_0
    PC16 is most influenced by: coupon_Coffee House
    PC17 is most influenced by: maritalStatus_Single
    PC18 is most influenced by: coupon_Restaurant(<20)
    PC19 is most influenced by: time_10AM
    PC20 is most influenced by: maritalStatus_Married partner
    PC21 is most influenced by: time_10PM
    PC22 is most influenced by: coupon_Carry out & Take away
    PC23 is most influenced by: coupon_Bar
    PC24 is most influenced by: weather_Sunny
    PC25 is most influenced by: occupation_Unemployed
    PC26 is most influenced by: time_2PM
    PC27 is most influenced by: coupon_Restaurant(20-50)
    PC28 is most influenced by: occupation_Computer & Mathematical
    PC29 is most influenced by: passanger_Partner
    PC30 is most influenced by: occupation_Sales & Related

Re-trained the basic models after applying PCA, to explore any improvement in performance. 

### Part 4 Summary and Evaluation of Model performance

The following tables summarize the results of Validation Accuracy,  Testing Accuracy  and AUC - Area Under the Curve of Models (with Data before and after PCA) 

From the tables it clear that KNN Classifier performs better than the rest of the models.(in both Categories - i.e, with PCA and without PCA)

The performance of the models was slightly better without PCA. 
One thing to note though, the models took comparatively lesser time to train with PCA.  

| Models with out PCA | Validation Accuracy|Testing Accuracy|AUC|
|-------|---------|-------|-----------|
|Logistic Regression with No Penalty|0.683|0.673|0.73|
|Lasso Logistic Regression|0.685|0.674|0.73|
|Ridge Logistic Regression|0.685|0.674|0.73|
|Decision Tree|0.684|0.686|0.73|
|Bernoulli Naive Bayes|0.657|0.655|0.70|
|**KNN Classifier**|**0.722**|**0.732**|**0.79**|
|Linear SVM|0.685|0.672|0.73|

| Models with PCA | Validation Accuracy|Testing Accuracy|AUC|
|-------|---------|-------|-----------|
|Logistic Regression with No Penalty|0.679|0.671|0.72|
|Lasso Logistic Regression|0.679|0.672|0.72|
|Ridge Logistic Regression|0.679|0.671|0.72|
|Decision Tree|0.650|0.648|0.68|
|Bernoulli Naive Bayes|0.657|0.655|0.68|
|**KNN Classifier**|**0.722**|**0.720**|**0.78**|
|Linear SVM|0.679|0.672|0.72|

## Link to Notebook:
[Explore Coupon Acceptance Factors by Customers](https://github.com/nbajam/BH-PCAIML-project/blob/main/bh_pcaiml_project_final.ipynb)

## Data Used:
[In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)

