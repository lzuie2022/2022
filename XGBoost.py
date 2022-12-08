import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler


#ss_y = StandardScaler()

path = "data.csv"
data = pd.read_csv(path, encoding = "ISO-8859-15",low_memory=False)

data_try = data[["feature_name"]]

    
    
X = data_try.drop(["correct"],axis = 1)
y  = data_try["correct"] 

import time
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
 
start = time.time()
 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
 
xgb = XGBClassifier(learning_rate = 0.7, 
                     gamma=0.1,
                     max_depth=9, 
                     n_estimators=300, 
                     min_child_weight=0.8, 
                     subsample=0.9,
                     colsample_bytree=0.9, 
                     objective= 'binary:logistic', 
                     nthread=4, 
                     seed=50,
                     reg_lambda=0.9,
                     reg_alpha=0.1 
                    )

#X_train = ss_y.fit_transform(X_train)   
#y_train = ss_y.fit_transform(y_train)   
xgb.fit(X_train, y_train)
  
acc = xgb.score(X_test, y_test)
  
print("train_acc", acc)
print("test_acc", xgb.score(X_train, y_train))
end = time.time()
print("time:" , end - start, 's') 

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
 
y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_proba[:,1], pos_label=1)
roc_auc = metrics.auc(fpr,tpr)
print("auc:",roc_auc)