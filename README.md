# BigDataProject

## Authors
[Marwan Ayadi](https://github.com/marwan-ayadi)
[Bicher Chammaa](https://github.com/bicher123)

## Abstract
The objective of this project is to analyze a dataset, discuss, and interpret the results. The dataset in question is called [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists?select=aug_train.csv). It contains 19158 data points, with 13 categorical and numerical features that will help classify the candidates. The goal is to determine whether a candidate will join the company or not using the different features and information about each candidate.   

## I. Introduction
### Context 
[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists) is a collection of data from a company that is active in Big Data and Data Science. They want to hire data scientists that successfully pass their courses. This company offers courses to potential employees in order to train them and facilitate their onboarding experience. This will help reduce the learning curve at the beginning of their work mandate and attract more potential employees.

### Objectives
Since a lot of people register for these courses, the company wants to find out if the candidates that sign up for the courses will eventually join the company. This will help the company reduce the cost and resources put into each candidate depending on whether or not they will be working with them. The objective of this study is to classify the different candidates' seriousness in working for the company. By using the candidates' credentials, experience, and demographics, we will predict the probability and interpret the factors associated with a candidate's decision to look for a new job or work for the company.

### Presentation of the Problem to Solve
Analyzing this dataset requires a supervised binary classification. One problem that we face with the data is that it is an imbalanced dataset. Since it is imbalanced the classifier might return a biased result towards the majority class. 

### Related Work.
Many notebooks have been posted concerning this dataset such as [Predict who will move to a new job](https://www.kaggle.com/khotijahs1/predict-who-will-move-to-a-new-job/notebook) by Siti Khotijah. In her analysis, she compares the different features that can influence a candidate's decision to move to a new job. She was able to find which features are noticeably affecting the classification of the candidates. By separating candidates according to their education level and comparing them using other sub-features, she was able to determine the classification of the candidates.

## II. Materials and Methods 
### The Dataset
The dataset in question comes in two separate files: one CSV file for the training data and one CSV file for the testing data. The Jabo Change of Data Scientists dataset provides 13 columns of data about a candidate, including gender, relevant experience, education level, city development index, etc. 

### Technologies and Algorithms
To analyze and classify this dataset, we will use the decision tree algorithm provided by Scikit-Learn. The decision tree algorithm is a tree structure in which each node represents a feature, each edge or branch represents a decision/value range, and each leaf is a label/outcome of the decision. There are different ways of determining the quality of the decision trees, namely the Gini index, and the Entropy.
