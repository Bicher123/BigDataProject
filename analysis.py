from collections import Counter
from numpy.lib import math
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import entropy,reciprocal
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics 

def read_files():
    train_set = pd.read_csv("./data/aug_train.csv")
    test_set = pd.read_csv("./data/aug_test.csv")
    answers = np.load('./data/jobchange_test_target_values.npy')

    df_answers = pd.DataFrame(answers)
    return train_set,test_set,df_answers

def vect_gender(data:str):
    col_options = {"Male": 0, "Female": 1, "Other": 2}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 3
   
def vect_relevent_experience(data:str):
    col_options = {"No relevent experience": 0, "Has relevent experience": 1}
    return col_options[data]

def vect_enrolled_university(data: str):
    col_options = {
        "no_enrollment": 0, "Part time course": 1, "Full time course": 2}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 3
    
def vect_education_level(data: str):
    col_options = {"Primary School": 0, "High School":1, "Graduate": 2, "Masters": 3, "Phd": 4}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 5

def vect_major_discipline(data: str):
    col_options = {"STEM":0, "Business Degree": 1, "Humanities": 2, "No Major": 3, "Other": 4, "Arts": 5}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #      return 6

def vect_experience(data):
    col_options = {"<1": 0, ">20": 21}
    for i in range(20):
        col_options[str(i+1)]=i+1
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #     return 22
    
def vect_company_size(data: str):
    col_options = {"<10": 0, "10/49+": 1, "50-99": 2, "100-500": 3, "500-999":4, "1000-4999": 5, "5000-9999":6, "10000+":7}
    if(data in col_options.keys()):
         return col_options[data]
    # else:
    #     return 8
def vect_company_type(data: str):
    col_options = {"Pvt Ltd": 0, "Funded Startup": 1, "Public Sector": 2, "Early Stage Startup": 3, "Other": 4, "NGO": 5}
    if(data in col_options.keys()):
        return col_options[data]
    # else:
    #     return 6

def vect_last_new_job(data: str):
    col_options = {"never": 0,"1": 1, "2": 2, "3": 3, "4": 4, ">4": 5}
    if(data in col_options.keys()):
        return col_options[data]
    # else:
    #     return 6


def vectorise_data(df_train, df_test, df_answer, isNull):
    # df_train = df_train.drop(df_train[(df_train.isnull().sum(axis=1) >2)].index)


    df_train['city'] = df_train['city'].str[5:].values.astype('int')
    df_train['gender'] = df_train.apply(lambda row: vect_gender(row['gender']), axis=1)
    df_train['relevent_experience'] = df_train.apply(lambda row: vect_relevent_experience(row['relevent_experience']), axis=1)
    df_train['enrolled_university'] = df_train.apply(lambda row: vect_enrolled_university(row['enrolled_university']), axis=1)
    df_train['education_level'] = df_train.apply(lambda row: vect_education_level(row['education_level']), axis=1)
    df_train['major_discipline'] = df_train.apply(lambda row: vect_major_discipline(row['major_discipline']), axis=1)
    df_train['experience'] = df_train.apply(lambda row: vect_experience(row['experience']), axis=1)
    df_train['company_size'] = df_train.apply(lambda row: vect_company_size(row['company_size']), axis=1)
    df_train['company_type'] = df_train.apply(lambda row: vect_company_type(row['company_type']), axis=1)
    df_train['last_new_job'] = df_train.apply(lambda row: vect_last_new_job(row['last_new_job']), axis=1)

    df_test['city'] = df_test['city'].str[5:].values.astype('int')
    df_test['gender'] = df_test.apply(lambda row: vect_gender(row['gender']), axis=1)
    df_test['relevent_experience'] = df_test.apply(lambda row: vect_relevent_experience(row['relevent_experience']), axis=1)
    df_test['enrolled_university'] = df_test.apply(lambda row: vect_enrolled_university(row['enrolled_university']), axis=1)
    df_test['education_level'] = df_test.apply(lambda row: vect_education_level(row['education_level']), axis=1)
    df_test['major_discipline'] = df_test.apply(lambda row: vect_major_discipline(row['major_discipline']), axis=1)
    df_test['experience'] = df_test.apply(lambda row: vect_experience(row['experience']), axis=1)
    df_test['company_size'] = df_test.apply(lambda row: vect_company_size(row['company_size']), axis=1)
    df_test['company_type'] = df_test.apply(lambda row: vect_company_type(row['company_type']), axis=1)
    df_test['last_new_job'] = df_test.apply(lambda row: vect_last_new_job(row['last_new_job']), axis=1)

    if not isNull:
        df_train['gender'] = df_train['gender'].fillna(round((df_train['gender'].mean())))
        df_train['enrolled_university'] = df_train['enrolled_university'].fillna(0)
        df_train['major_discipline'] = df_train['major_discipline'].fillna(round((df_train['major_discipline'].mean())))
        df_train['company_size'] = df_train['company_size'].fillna(round((df_train['company_size'].mean())))
        df_train['company_type'] = df_train['company_type'].fillna(round((df_train['company_type'].mean())))
        df_train['experience'] = df_train['experience'].fillna(round((df_train['experience'].mean())))
        df_train['last_new_job'] = df_train['last_new_job'].fillna(round((df_train['last_new_job'].mean())))
        df_train['education_level'] = df_train['education_level'].fillna(round((df_train['education_level'].mean())))
        
        df_test['gender'] = df_test['gender'].fillna(round((df_test['gender'].mean())))
        df_test['enrolled_university'] = df_test['enrolled_university'].fillna(0)
        df_test['major_discipline'] = df_test['major_discipline'].fillna(round((df_test['major_discipline'].mean())))
        df_test['company_size'] = df_test['company_size'].fillna(round((df_test['company_size'].mean())))
        df_test['company_type'] = df_test['company_type'].fillna(round((df_test['company_type'].mean())))
        df_test['experience'] = df_test['experience'].fillna(round((df_test['experience'].mean())))
        df_test['last_new_job'] = df_test['last_new_job'].fillna(round((df_test['last_new_job'].mean())))
        df_test['education_level'] = df_test['education_level'].fillna(round((df_test['education_level'].mean())))

    
    print(df_train.head())
    return df_train, df_test, df_answer

def plot_target_distribution(df_train):
    y =[(df_train.target == 0).sum(), (df_train.target == 1).sum()]
    x =['0', '1']
    x_axis = np.array(x)
    y_axis = np.array(y)
    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=90)
    plt.show()

def calculate_correlations(df_train):

    corrMatrix = df_train.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    columns = list(df_train.columns)
    correlations = []
    
    # for index, col in enumerate(columns):
    #     if index < len(columns)-2: 
    #         for other_col in columns[index+1:]:
    #             correlations.append((col, other_col, df_train[col].corr(df_train[other_col])))
    # print(correlations)
    return correlations
def calculate_entropy(df_train):
    entropies = []
    for col in list(df_train.columns):
        c = len(df_train[col].unique().tolist())
        pd_series = df_train[col].squeeze()
        counts = pd_series.value_counts()
        e = entropy(counts)
        entropies.append((col, e, c, math.log(c,2),e/math.log(c,2)))
    x = []
    y = []

    for ent in entropies:
        x.append(ent[0])
        y.append(ent[4])

   
    x_axis = np.array(x)
    y_axis = np.array(y)


    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=90)
    plt.show()


a,b,c=read_files()
# // false is no null in columns
print("Any missing sample in training set:",a.isnull().values.any())
df_train, df_test, df_answer = vectorise_data(a,b,c , False)
print("Any missing sample in training set after preparaton:",df_train.isnull().values.any())
df_train = df_train.drop(['enrollee_id', 'city','last_new_job', 'company_size'], axis=1)
df_test = df_test.drop(['enrollee_id', 'city','last_new_job', 'company_size'], axis=1)
plot_target_distribution(df_train)
calculate_correlations(df_train)
calculate_entropy(df_train)
df_target = df_train['target']
df_train = df_train.drop(['target'], axis=1)

param_dist = {"max_depth": [5,10,15,20,25,50,None],
                "max_features": [ 2,4,6, 8,'auto', 'sqrt', 'log2', None],
                "min_samples_leaf": [ 2, 4, 6, 8, 10,12],
                "criterion": ["gini", "entropy"]}

counter = Counter(df_target)
print('Before',counter)
sm = SMOTE(sampling_strategy = .9)

df_train, df_target = sm.fit_resample(df_train, df_target)
# ncr= NeighbourhoodCleaningRule()
# df_train, df_target = ncr.fit_resample(df_train, df_target)
counter = Counter(df_target)
print('After',counter)
               
tree = DecisionTreeClassifier()
noDt = tree.fit(df_train,df_target)
nopred = noDt.predict(df_test)
print(metrics.classification_report(df_answer, nopred))
# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=50)

# Fit it to the data
tree_cv.fit(df_train,df_target)
prediction = tree_cv.predict(df_test)
print(tree_cv.best_params_)

print(metrics.classification_report(df_answer, prediction))
disp = plot_confusion_matrix(tree_cv, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
plt.show()

print("=====================================================================")
print("Random Forest classification:")
rf = RandomForestClassifier()
no_hp_rf = rf.fit(df_train,df_target)
no_hp_rf_pred = no_hp_rf.predict(df_test)
print(metrics.classification_report(df_answer, no_hp_rf_pred))

rf_cv = RandomizedSearchCV(rf, param_dist, cv=10)
rf_cv.fit(df_train,df_target)
prediction_rf = rf_cv.predict(df_test)
print(rf_cv.best_params_)

print(metrics.classification_report(df_answer, prediction_rf))
disp = plot_confusion_matrix(rf_cv, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
plt.show()
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# 


# pipeline = Pipeline(steps = [('smote', sm),('under',rus)])


# #over and undersampling the train dataset using SMOTE + RandomUnderSampler
# X_train_smrus, y_train_smrus = pipeline.fit_resample(X_train, y_train)

# counter = Counter(y_train_smrus)
# print('After',counter)

    # # Instantiate a Decision Tree classifier: tree
    # tree = DecisionTreeClassifier()

    # # Instantiate the RandomizedSearchCV object: tree_cv
    # tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

    # # Fit it to the data
    # tree_cv.fit(X,y)

    # # Print the tuned parameters and score
    # print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    # print("Best score is {}".format(tree_cv.best_score_))

    # test_set["result"] = df_answers

    # results = df_answers[0]
    # print(results.head())
    # print(train_set.shape)
    # print(train_set.head())

    #drops the target column. axis is 1 to drop the column
    # X_train = train_set.drop(['target', 'city'], axis=1)
    # y_train = train_set['target']

    # X_test = test_set.drop('city', axis=1)
    # y_test = df_answers

    # classifier = DecisionTreeClassifier()
    # classifier.fit(X_train, y_train)

    # y_pred = classifier.predict(X_test)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    

#     from imblearn.over_sampling import SMOTE
# from collections import Counter

