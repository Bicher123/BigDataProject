from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

train_set = pd.read_csv("./data/aug_train.csv")
test_set = pd.read_csv("./data/aug_test.csv")
answers = np.load('./data/jobchange_test_target_values.npy')

df_answers = pd.DataFrame(answers)
# test_set["result"] = df_answers

# results = df_answers[0]
# print(results.head())
# print(train_set.shape)
# print(train_set.head())

#drops the target column. axis is 1 to drop the column
X_train = train_set.drop(['target', 'city'], axis=1)
y_train = train_set['target']

X_test = test_set.drop('city', axis=1)
y_test = df_answers

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
    