from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

def balanceDataSet(df_train, df_target):
    #  Balancing data using SMOTE
    counter = Counter(df_target)
    print('Before balancing ',counter)
    sm = SMOTE(sampling_strategy = .9)
    df_train, df_target = sm.fit_resample(df_train, df_target)
    counter = Counter(df_target)
    print('After balancing ',counter)

    y =[counter[0.0], counter[1.0]]
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
    return df_train, df_target