from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd


def randomForestClassification(df_train,df_target, df_answer, df_test):
    print("=====================================================================")
    print("Random Forest classification:")
    # random forest classification without hyperparameters
    rf = RandomForestClassifier()
    # training
    no_hp_rf = rf.fit(df_train,df_target)
    # predict test
    no_hp_rf_pred = no_hp_rf.predict(df_test)
    # predict train
    no_hyp_pred_train = no_hp_rf.predict(df_train)

    print("train base")
    print(classification_report(df_target, no_hyp_pred_train))
    print("test base")
    print(classification_report(df_answer, no_hp_rf_pred))
    disp = plot_confusion_matrix(no_hp_rf, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    plt.show()

    #  possible parameters for the RandomSearchCV hyperparameter process
    params = { 'n_estimators':[i for i in range(10,100,10)],
                "max_depth": [2, 4, 8, 12],
                "max_features": [ 2,5,7,'auto', 'log2',None],
                "min_samples_leaf": [ 1, 3, 5],
                "min_samples_split": [2, 6, 12, 24, 48, 96],
                "criterion": ["gini", "entropy"]}

    # random forest classification with hyperparameters               
    rf_cv = RandomizedSearchCV(rf, params, cv=10)
   
    #  training
    hp_rf = rf_cv.fit(df_train,df_target) 
    print(rf_cv.best_params_)
    # predict test
    prediction_rf = hp_rf.predict(df_test)
    # predict train
    pred_train = hp_rf.predict(df_train)

    print("train")
    print(classification_report(df_target, pred_train))
    print("test")
    print(classification_report(df_answer, prediction_rf))
    disp = plot_confusion_matrix(rf_cv, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    plt.show()

    print("random")
    data = np.random.randint(0,2,size=len(df_target))
    # df_rand = pd.DataFrame(data, columns=['random_numbers'])
    rf_cv.fit(df_train,data)
    rand_pred = rf_cv.predict(df_test)
    print(classification_report(df_answer, rand_pred))