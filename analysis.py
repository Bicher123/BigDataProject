

from numpy.lib import math
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NeighbourhoodCleaningRule
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import entropy,reciprocal
from sklearn.ensemble import RandomForestClassifier
from decision_tree import decisionTreeClassification
from random_forest import randomForestClassification


from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics 
from vectorization import vectorise_data
from balance import balanceDataSet

def read_files():
    train_set = pd.read_csv("./data/aug_train.csv")
    test_set = pd.read_csv("./data/aug_test.csv")
    answers = np.load('./data/jobchange_test_target_values.npy')

    df_answers = pd.DataFrame(answers)
    return train_set,test_set,df_answers


def plot_target_distribution(df_train):
    y =[(df_train.target == 0).sum(), (df_train.target == 1).sum()]
    x =['0', '1']
     # x-coordinates of left sides of bars 
    left = [1, 2] 
    
    x_axis = np.array(x)
    y_axis = np.array(y)
    
    plt.xticks(rotation=90)
    bar=plt.bar(left, y_axis, tick_label = x_axis, 
            width = 0.8, color = ['blue', 'green']) 
    # show bar values on top
    for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
    plt.show()

def calculate_correlations(df_train):

    corrMatrix = df_train.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

    return []

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


#  read csv file 
train_set,test,answer=read_files()

print("Any missing sample in training set:",train_set.isnull().values.any())
df_train, df_test, df_answer = vectorise_data(train_set,test,answer , False)
print("Any missing sample in training set after preparaton:",df_train.isnull().values.any())


plot_target_distribution(df_train)
calculate_correlations(df_train)
calculate_entropy(df_train)

# drop columns after anlysing the correlation matrice and the entropy chart
df_target = df_train['target']
df_train = df_train.drop(['enrollee_id', 'city','last_new_job', 'target', 'enrolled_university'], axis=1)
df_test = df_test.drop(['enrollee_id', 'city','last_new_job','enrolled_university'], axis=1)

df_train, df_target = balanceDataSet(df_train, df_target)
df_train['target'] = df_target
df_train, validation = train_test_split(df_train, test_size=0.1)
df_target = df_train['target']
df_train = df_train.drop(['target'], axis=1)
validation_answer = validation['target']
validation = validation.drop(['target'], axis=1)

# decision tree classification
decisionTreeClassification(df_train,df_target, df_answer, df_test, validation, validation_answer)
randomForestClassification(df_train,df_target, df_answer, df_test, validation, validation_answer)
