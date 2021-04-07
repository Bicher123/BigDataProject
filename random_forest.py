from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


def randomForestClassification(df_train,df_target, df_answer, df_test):
    print("=====================================================================")
    print("Random Forest classification:")
    # random forest classification without hyperparameters
    rf = RandomForestClassifier()
    no_hp_rf = rf.fit(df_train,df_target)
    no_hp_rf_pred = no_hp_rf.predict(df_test)
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
    hp_rf = rf_cv.fit(df_train,df_target)
    prediction_rf = rf_cv.predict(df_test)
    print(rf_cv.best_params_)

    print(classification_report(df_answer, prediction_rf))
    disp = plot_confusion_matrix(rf_cv, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    plt.show()
