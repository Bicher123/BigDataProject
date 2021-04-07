from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV



def decisionTreeClassification(df_train,df_target, df_answer, df_test):
    print("=====================================================================")
    print("Decision Tree classification:")

    # Decision tree without hyperparameters               
    tree = DecisionTreeClassifier()
    nohyp = tree.fit(df_train,df_target)
    noHypPred = nohyp.predict(df_test)
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

    # Decision tree with hyperparameters         
    rand_tree.fit(df_train,df_target)
    prediction = rand_tree.predict(df_test)
    print(rand_tree.best_params_)
    print(classification_report(df_answer, prediction))
    disp = plot_confusion_matrix(rand_tree, df_test, df_answer,display_labels=['0','1'],cmap=plt.cm.Blues,normalize=None)
    plt.show()