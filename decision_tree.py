from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd



def decisionTreeClassification(df_train,df_target, df_answer, df_test, validation, validation_answer):
    print("=====================================================================")
    print("Decision Tree classification:")

    # Decision tree without hyperparameters               
    tree = DecisionTreeClassifier()
    # training
    nohyp = tree.fit(df_train,df_target)
    # predict test
    noHypPred = nohyp.predict(df_test)
    # predict validation
    no_hyp_pred_val = nohyp.predict(validation)
    # predict validation
    no_hyp_pred_train = nohyp.predict(df_train)

    print("train base ")
    print(classification_report(df_target, no_hyp_pred_train))
    print("validation base ")
    print(classification_report(validation_answer, no_hyp_pred_val))
    print("test base")
    print(classification_report(df_answer, noHypPred))
    disp = plot_confusion_matrix(nohyp, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    plt.show()

    #  possible parameters for the RandomSearchCV hyperparameter process
    params = { "max_depth": [2, 4, 8, 12],
                "max_features": [ 2,5,7,'auto', 'log2',None],
                "min_samples_leaf": [ 1, 3, 5],
                "min_samples_split": [2, 6, 12, 24, 48, 96],
                "splitter": ['best', 'random'],
                "criterion": ["gini", "entropy"]}

    rand_tree = RandomizedSearchCV(tree, params, cv=30)
    
    # training        
    rand_tree.fit(df_train,df_target)
    print(rand_tree.best_params_)
    # predict test
    prediction = rand_tree.predict(df_test)
    # predict validation
    pred_val = rand_tree.predict(validation)
    # predict train
    pred_train = rand_tree.predict(df_train)
   
    print("train hyper")
    print(classification_report(df_target, pred_train))
    print("validation hyper")
    print(classification_report(validation_answer, pred_val))
    print("test hyper")
    print(classification_report(df_answer, prediction))
    disp = plot_confusion_matrix(rand_tree, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    plt.show()

    print("random")
    data = np.random.randint(0,2,size=len(df_target))
    df_rand = pd.DataFrame(data, columns=['random_numbers'])
    rand_tree.fit(df_train,df_rand)
    rand_pred = rand_tree.predict(df_test)
    print(classification_report(df_answer, rand_pred))



