import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM,afm,xdeepfm,fnn,fibinet,pnn
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as KK
from tensorflow.keras import losses
#from tensorflow.keras.models import Model
#from deepctr.inputs import  SparseFeat, DenseFeat,get_feature_names
from deepctr.feature_column import SparseFeat,DenseFeat,get_feature_names
from sklearn.metrics import roc_auc_score
import time
from deepctr.models import AutoInt

start = time.time()

def auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return KK.sum(s, axis=0)  
    
def binary_PTA(y_true, y_pred, threshold=KK.variable(value=0.5)):  
    y_pred = KK.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = KK.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = KK.sum(y_pred * y_true)  
    return TP/P
    
def binary_PFA(y_true, y_pred, threshold=KK.variable(value=0.5)):  
    y_pred = KK.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = KK.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = KK.sum(y_pred - y_pred * y_true)  
    return FP/N 

data = pd.read_csv('data.csv')
cols = ["cols_name"]
data = data.loc[:,cols].dropna()

sparse_features = ['sparse_features_name']
dense_features = []
target = ['correct']

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)]
dense_feature_columns = [DenseFeat(feat, 1)
                      for feat in dense_features]

dnn_feature_columns = sparse_feature_columns + dense_feature_columns
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# data = pd.read_csv('/content/drive/My Drive/app/dkt_DFM/data.csv')
# cols = ['user','item','skill','wins','fails']

# sparse_features = ['skill','wins','fails']
train, valid = train_test_split(data, test_size=0.1)

train_model_input = {name:train[name].values for name in feature_names}
valid_model_input = {name:valid[name].values for name in feature_names}
model = AutoInt(linear_feature_columns,dnn_feature_columns,task='binary',dnn_dropout=0.5,l2_reg_dnn=1e-2)
adam = Adam(lr=5e-6,epsilon=1e-10)

model.compile(optimizer="adam",loss = losses.binary_crossentropy,metrics=[auc,'acc'])

history = model.fit(train_model_input, train[target].values,
                    batch_size=512, epochs=50, verbose=2, validation_split=0.1)

score = model.evaluate(valid_model_input,valid[target].values)
print(model.metrics_names)
print('valid score = ' ,score)

end = time.time()
print("time:" , end - start, 's') 