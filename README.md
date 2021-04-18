# BigDataProject

## Authors
[Marwan Ayadi](https://github.com/marwan-ayadi)  <br />
[Bicher Chammaa](https://github.com/bicher123)

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the require libraries.
```bash
pip3 install -r requirements.txt
```
### Running the project

```bash
python analysis.py
```

## Abstract
The objective of this project is to analyze a dataset, discuss, and interpret the results. The dataset in question is called [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists?select=aug_train.csv). It contains 19158 data points, with 13 categorical and numerical features that will help classify the candidates. The goal is to determine whether a candidate will join the company or not using the different features and information about each candidate.   

## I. Introduction
### Context 
[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists) is a collection of data from a company that is active in Big Data and Data Science. They want to hire data scientists that successfully pass their courses. This company offers courses to potential employees in order to train them and facilitate their onboarding experience. This will help reduce the learning curve at the beginning of their work mandate and attract more potential employees.

### Objectives
Since a lot of people register for these courses, the company wants to find out if the candidates that sign up for the training will eventually join the company. This will help the company reduce the costs and resources put into each candidate depending on whether they will be working with them. The objective of this study is to classify the different candidates' seriousness in working for the company. By using the candidates' credentials, experience, and demographics, we will predict the probability and interpret the factors associated with a candidate's decision to look for a new job or work for the company.

### Presentation of the Problem to Solve
Analyzing this dataset will be performed using supervised binary classification. One problem that we face with the data is that it is an imbalanced dataset. Since it is imbalanced the classifier might return a biased result towards the majority class.

### Related Work.
Many notebooks have been posted concerning this dataset such as [Predict who will move to a new job](https://www.kaggle.com/khotijahs1/predict-who-will-move-to-a-new-job/notebook) by Siti Khotijah. In her analysis, she compares the different features that can influence a candidate's decision to move to a new job. She was able to find which features are noticeably affecting the classification of the candidates. By separating candidates according to their education level and comparing them using other sub-features, she was able to determine the classification of the candidates.

## II. Materials and Methods 
### The Dataset
The dataset in question comes in three separate files: one CSV file for the training data, one CSV file for the testing data, and one NumPy file which contains the test answers. The Job Change of Data Scientists dataset provides 13 columns of features about a candidate, including gender, relevant experience, education level, city development index, etc. This list of features is provided below in more detail.

|Features|Possible Values|
|--------|---------------|
|enrollee_id|(numbers)|
|city|City_(numbers)|
|city_development_index|(numbers)|
|gender|Male, Female, Other, NaN|
|relevant_experience|No relevant experience, Has relevant experience|
|enrolled_university|no_enrollment, Part time course, Full time course, NaN|
|education_level|Primary School, High School, Graduate, Masters, Phd, NaN|
|major_discipline|STEM, Business Degree, Humanities, No Major, Other, Arts, NaN|
|experience|(Numbers), <1, >20, NaN|
|company_size|<10, 10/49+, 50-99, 100-500, 500-999, 1000-4999, 5000-9999, 10000+, NaN|
|company_type|Pvt Ltd, Funded Startup, Public Sector, Early Stage Startup, Other, NGO, NaN|
|last_new_job|never,1, 2, 3, 4, >4, NaN|
|training_hours|(numbers)|

Finally, the last column is the target column, which is broken down into the following:
||Target|Definition|
|---|---|---|
|Positive|0|Will not look for another job|
|Negative|1|Will look for another job|

Most of the features are represented with string expressions which will need some processing to be able to be used in the later algorithms.
Before using any algorithm, we analyzed the data to know more about the dataset. The dataset has 20 733 missing fields which represent only 0.08% of the total number of fields. However, these missing fields are still relevant because the missing data affects 53.26% of the total rows in the dataset.

### Data Imbalance
Then we analyzed the target distribution to determine the breakdown of the dataset’s target class. As we can see in figure 1, the dataset is imbalanced with 14381 “0” target (~75% of the dataset) and 4777 “1” target (~25% of the dataset). 

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/count.png) <br />
*Figure 1 Target distribution in the training dataset*


We chose to address the dataset imbalance by applying the Synthetic Minority Oversampling Technique or SMOTE to the data. By creating new data points, we made sure that all the original data points remained through the analysis.   When generating new data points, we chose a sampling strategy of 0.9, meaning that the ratio of the minority class over the majority class is equal to 0.9. This way, we made sure that the dataset is more balanced without having to create too many new data points. As a result, the updated distribution can be seen below:

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/count_after.png)  <br />
*Figure 2 Updated target distribution in the training dataset after applying SMOTE*

### Data Processing
Since most of the columns’ values are of type string, we decided to map the string values to integers to be able to process them with the scikit-learn library. To do that, we created dictionaries for each column containing all the possible values of that column to be able to map all the values to integers using the apply method (see example in figure 3). Only the city number did not require a dictionary since we just had to remove the substring “city_” and cast the rest into integers. As for the missing data, we first tried to replace them by considering them as another option value and mapped them to a number (see else section in figure 2). Another attempt was to replace all the missing field values by the average of the column they belong to. We chose the average to ensure that the new values were not outliers. The average value assignment to the missing fields resulted in better results when comparing it with other options.

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/vect.PNG)  <br />
*Figure 3 Gender vectorization*

### Feature Analysis
To determine which feature will be used in the classification, we computed the correlation matrix between columns (see figure 4).

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/correlation.png)  <br />
*Figure 4 Correlation matrix of the initial dataset*

We decided to drop the columns that correlate higher than or equal to 0.35. When choosing between two correlated columns, we used entropy measurements to determine the best choice to keep (see figure 5).
We calculated the entropy using the spicy.stat library. We then normalized all the entropy values by dividing the entropy result by the log base 2 of the length of unique possible outcomes of the column.
Higher entropy means less information can be concluded from the column.  As a result, we decided to remove the enrollee_id column since it has high entropy. Also, when examining the values of that column, we can see that each row has a unique value, and it is not relevant to the classification.

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/entropy.png)  <br />
*Figure 5 Entropy chart by feature*

Columns dropped: 'enrollee_id', 'city','last_new_job','enrolled_university'


### Technologies and Algorithms
To analyze and classify this dataset, we will use the decision tree classification and the random forest classification, both provided by the Scikit-Learn library. 
The decision tree algorithm is a tree structure in which each node represents a feature, each edge or branch represents a decision/value range, and each leaf is a label/outcome of the decision.

The random forest is similar to the decision tree but uses multiple trees by selecting a random sample from the training data and the features. After multiple trees are constructed, each tree provides its classification for a data point and the target value with the most votes gets classified as such. This algorithm should theoretically be more effective and should reduce the overfitting effect from the decision tree.

For comparison purposes, we decided to run each classification algorithm four times. First, we ran the decision tree by predicting the training set (which should be almost 100% accuracy) to validate that the algorithm trained correctly. Secondly, we ran the algorithm with the default parameters provided by the Scikit-Learn DecisionTreeClassifier and RandomForestClassifier (base) on the test set to compare the result with the hyper parameterized algorithm. Then, we ran the algorithm after hyperparameter tuning (better) on the test set. Finally, we ran the algorithm by training it with a randomly generated binary target using the same hyperparameters used before. This last run will help us determine if the algorithm is really working or if it is randomly predicting the target values.

As for the hyperparameter tuning, we used the random search cross-validation for both algorithms which selects the best parameters from multiple random parameter values.
The random search cross-validation has a probability of selecting a value in the property array of 1 over the length of the array. For example, this amounts to a probability of ¼ to select a value from the max_depth property in figure 6. The array provided in each property represents the possible values available in the random search cross-validation that we estimated to be the most meaningful values for our project. 

Here in the decision tree, we do 30 cross-validations.

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/parameters_dt.PNG)  <br />
*Figure 6 Possible parameters for the decision tree*

|Parameter|Description|
|---------|-----------|
|max_depth|	The maximum depth of the tree|
|max_features|	The number of features to consider when determining the best split at a node|
|min_samples_split|	The minimum number of samples required to split an internal node|
|min_samples_leaf|	The minimum number of samples required to be at a leaf node|
|splitter|	The strategy to use when determining the next split at a node|
|criterion|	The function to use when evaluating the quality of a split|

For the random forest, we do 10 (less than decision tree, due to time of execution which is longer)

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/parameters_rf.PNG)  <br />
*Figure 7 Possible parameters for the random forest*

|Parameter|Description|
|---------|-----------|
|n_estimators|The number of trees in the random forest|
|max_depth|	The maximum depth of the tree|
|max_features|	The number of features to consider when determining the best split at a node|
|min_samples_split|	The minimum number of samples required to split an internal node|
|min_samples_leaf|	The minimum number of samples required to be at a leaf node|
|criterion|	The function to use when evaluating the quality of a split|

## Results
Prediction on random target results for both algorithms:
<table>
  <tr>
    <th></th>
    <th colspan="3">Decision Tree With hyperparameters</th>
    <th colspan="3">Random Forest With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.64</td>
    <td>0.01</td>
    <td>0.01</td>
    <td>0.73</td>
    <td>0.51</td>
    <td>0.60</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.27</td>
    <td>0.99</td>
    <td>0.42</td>
    <td>0.27</td>
    <td>0.50</td>
    <td>0.35</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.54</td>
    <td>0.27</td>
    <td>0.12</td>
    <td>0.61</td>
    <td>0.51</td>
    <td>0.53</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.27</td>
    <td colspan="3">0.51</td>
  </tr>
</table>

### Decision Tree Classification
Hyperparameters:
```
{'splitter': 'best', 'min_samples_split': 96, 'min_samples_leaf': 5, 'max_features': 7, 'max_depth': 12, 'criterion': 'entropy'}
```
Prediction on the training set:
<table>
  <tr>
    <th></th>
    <th colspan="3">Without hyperparameters</th>
    <th colspan="3">With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.99</td>
    <td>1.00</td>
    <td>0.99</td>
    <td>0.85</td>
    <td>0.79</td>
    <td>0.82</td>
  </tr>
  <tr>
    <th>1</th>
    <td>1.00</td>
    <td>0.99</td>
    <td>0.99</td>
    <td>0.78</td>
    <td>0.84</td>
    <td>0.81</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.99</td>
    <td>0.99</td>
    <td>0.99</td>
    <td>0.82</td>
    <td>0.81</td>
    <td>0.81</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.99</td>
    <td colspan="3">0.81</td>
  </tr>
</table>

Prediction on the test set:
<table>
  <tr>
    <th></th>
    <th colspan="3">Without hyperparameters</th>
    <th colspan="3">With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.77</td>
    <td>0.84</td>
    <td>0.81</td>
    <td>0.81</td>
    <td>0.87</td>
    <td>0.84</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.44</td>
    <td>0.34</td>
    <td>0.38</td>
    <td>0.57</td>
    <td>0.45</td>
    <td>0.51</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.68</td>
    <td>0.71</td>
    <td>0.69</td>
    <td>0.75</td>
    <td>0.76</td>
    <td>0.75</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.71</td>
    <td colspan="3">0.76</td>
  </tr>
</table>

Prediction on the validation set:
<table>
  <tr>
    <th></th>
    <th colspan="3">Without hyperparameters</th>
    <th colspan="3">With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.79</td>
    <td>0.77</td>
    <td>0.78</td>
    <td>0.84</td>
    <td>0.77</td>
    <td>0.81</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.74</td>
    <td>0.76</td>
    <td>0.75</td>
    <td>0.80</td>
    <td>0.80</td>
    <td>0.79</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.76</td>
    <td>0.76</td>
    <td>0.76</td>
    <td>0.80</td>
    <td>0.80</td>
    <td>0.80</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.76</td>
    <td colspan="3">0.80</td>
  </tr>
</table>

### Random Forest Classification 
Hyperparameters:
```
{'n_estimators': 30, 'min_samples_split': 24, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 12, 'criterion': 'entropy'}
```
Prediction on the training set:

<table>
  <tr>
    <th></th>
    <th colspan="3">Without hyperparameters</th>
    <th colspan="3">With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>1.00</td>
    <td>1.00</td>
    <td>0.99</td>
    <td>0.87</td>
    <td>0.81</td>
    <td>0.84</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.99</td>
    <td>1.00</td>
    <td>0.99</td>
    <td>0.81</td>
    <td>0.86</td>
    <td>0.83</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.99</td>
    <td>0.99</td>
    <td>0.99</td>
    <td>0.84</td>
    <td>0.84</td>
    <td>0.84</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.99</td>
    <td colspan="3">0.84</td>
  </tr>
</table>

Prediction on the test set:
<table>
  <tr>
    <th></th>
    <th colspan="3">Decision Tree With hyperparameters</th>
    <th colspan="3">Random Forest With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.79</td>
    <td>0.92</td>
    <td>0.85</td>
    <td>0.81</td>
    <td>0.90</td>
    <td>0.85</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.61</td>
    <td>0.33</td>
    <td>0.43</td>
    <td>0.61</td>
    <td>0.43</td>
    <td>0.50</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.74</td>
    <td>0.76</td>
    <td>0.73</td>
    <td>0.76</td>
    <td>0.77</td>
    <td>0.76</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.66</td>
    <td colspan="3">0.77</td>
  </tr>
</table>

Prediction on the validation set:
<table>
  <tr>
    <th></th>
    <th colspan="3">Decision Tree With hyperparameters</th>
    <th colspan="3">Random Forest With hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.83</td>
    <td>0.85</td>
    <td>0.84</td>
    <td>0.85</td>
    <td>0.80</td>
    <td>0.82</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.82</td>
    <td>0.80</td>
    <td>0.83</td>
    <td>0.78</td>
    <td>0.84</td>
    <td>0.81</td>
  </tr>
  <tr>
    <th>Weighted Average</th>
    <td>0.83</td>
    <td>0.83</td>
    <td>0.83</td>
    <td>0.82</td>
    <td>0.82</td>
    <td>0.82</td>
  </tr>
  <tr>
    <th>Accuracy</th>
    <td colspan="3">0.83</td>
    <td colspan="3">0.82</td>
  </tr>
</table>

After performing the different runs on each supervised classifier, we can notice the difference that tuning the hyperparameters provides on the results. The purpose of adjusting those parameters was to ensure that the results obtained were the best possible results as well as to ensure that we avoid overfitting the dataset. 

As a result, we noticed that, in the case of the prediction on the training set, changing the parameters reduced the F1-score from 0.99 to 0.81, and 0.99 to 0.84 for the decision tree and the random forest classifiers, respectively. For the results of the validation and test set, we noticed that the F1-score received an increase after tuning the hyperparameters for both algorithms.

### Comparison

<table>
  <tr>
    <th></th>
    <th colspan="4">Prediction on the test set with hyperparameters</th>
  </tr>
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <th>Decision Tree</th>
    <td>0.75</td>
    <td>0.76</td>
    <td>0.75</td>
    <td>0.76</td>
  </tr>
  <tr>
    <th>Random Forest</th>
    <td>0.76</td>
    <td>0.77</td>
    <td>0.76</td>
    <td>0.77</td>
  </tr>
</table>

When comparing the two algorithms, we found that the Random Forest classifier always performed slightly better than the Decision Tree classifier. This can be attributed to the majority voting implemented within the random forest, in which multiple trees vote on the classification of a data point, as opposed to the decision tree which only relies on the classification of a single tree.

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/better_dt3.png)  <br />
*Figure 8 Confusion matrix for the decsion tree's best run*

![alt text](https://github.com/Bicher123/BigDataProject/blob/master/results/better_rand3.png)  <br />
*Figure 9 Confusion matrix for the random forest’s best run*

## Discussion
The results obtained from the eight different iterations above show that hyperparameter tuning reduced overfitting. This can be confirmed by the fact that the results of the prediction on the validation are closes to the results obtained on the test set after hyperparameter tuning.

We can see in the confusion matrices above the difference between a run with and without hyperparameter tuning. After changing the hyperparameters, we see an increase in True Negatives (TN) and True Positives (TP). 

Also, when comparing the Decision Tree and Random Forest classifiers, we see very similar results, but the Random Forest classifier always has a higher F1-score. This is because the random forest uses a large number of decision trees that decide the target value by majority vote.

Since Random Forest and decision tree are similar, we could use another algorithm such as kNN classification which may help us validate our results and see if we could potentially have higher metrics.
