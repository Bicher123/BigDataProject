from collections import Counter
from imblearn.over_sampling import SMOTE

def balanceDataSet(df_train, df_target):
    #  Balancing data using SMOTE
    counter = Counter(df_target)
    print('Before balancing ',counter)
    sm = SMOTE(sampling_strategy = .9)
    df_train, df_target = sm.fit_resample(df_train, df_target)
    counter = Counter(df_target)
    print('After balancing ',counter)
    return df_train, df_target