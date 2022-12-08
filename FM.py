import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM,NFM,afm,xdeepfm,fnn,fibinet,pnn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as KK
from tensorflow.keras import losses
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

class FM(six.with_metaclass(ABCMeta, _BasePoly)):

    @abstractmethod
    def __init__(self, degree=2, loss='squared', n_components=2, alpha=1,
                 beta=1, tol=1e-6, fit_lower='explicit', fit_linear=True,
                 warm_start=False, init_lambdas='ones', max_iter=10000,
                 verbose=False, random_state=None):
        self.degree = degree
        self.loss = loss
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.fit_lower = fit_lower
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.init_lambdas = init_lambdas
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def _augment(self, X):
        # for factorization machines, we add a dummy column for each order.

        if self.fit_lower == 'augment':
            k = 2 if self.fit_linear else 1
            for _ in range(self.degree - k):
                X = add_dummy_feature(X, value=1)
        return X

    def fit(self, X, y):
        """Fit factorization machine to training data.

        Parameters
        ----------
        X : array-like or sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : Estimator
            Returns self.
        """
        if self.degree > 3:
            raise ValueError("FMs with degree >3 not yet supported.")

        X, y = self._check_X_y(X, y)
        X = self._augment(X)
        n_features = X.shape[1]  # augmented
        X_col_norms = row_norms(X.T, squared=True)
        dataset = get_dataset(X, order="fortran")
        rng = check_random_state(self.random_state)
        loss_obj = self._get_loss(self.loss)

        if not (self.warm_start and hasattr(self, 'w_')):
            self.w_ = np.zeros(n_features, dtype=np.double)

        if self.fit_lower == 'explicit':
            n_orders = self.degree - 1
        else:
            n_orders = 1

        if not (self.warm_start and hasattr(self, 'P_')):
            self.P_ = 0.01 * rng.randn(n_orders, self.n_components, n_features)

        if not (self.warm_start and hasattr(self, 'lams_')):
            if self.init_lambdas == 'ones':
                self.lams_ = np.ones(self.n_components)
            elif self.init_lambdas == 'random_signs':
                self.lams_ = np.sign(rng.randn(self.n_components))
            else:
                raise ValueError("Lambdas must be initialized as ones "
                                 "(init_lambdas='ones') or as random "
                                 "+/- 1 (init_lambdas='random_signs').")

        y_pred = self._get_output(X)

        converged, self.n_iter_ = _cd_direct_ho(
            self.P_, self.w_, dataset, X_col_norms, y, y_pred,
            self.lams_, self.degree, self.alpha, self.beta, self.fit_linear,
            self.fit_lower == 'explicit', loss_obj, self.max_iter,
            self.tol, self.verbose)
        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, X):
        y_pred = _poly_predict(X, self.P_[0, :, :], self.lams_, kernel='anova',
                               degree=self.degree)

        if self.fit_linear:
            y_pred += safe_sparse_dot(X, self.w_)

        if self.fit_lower == 'explicit' and self.degree == 3:
            # degree cannot currently be > 3
            y_pred += _poly_predict(X, self.P_[1, :, :], self.lams_,
                                    kernel='anova', degree=2)

        return y_pred

    def _predict(self, X):
        if not hasattr(self, "P_"):
            raise NotFittedError("Estimator not fitted.")
        X = check_array(X, accept_sparse='csc', dtype=np.double)
        X = self._augment(X)
        return self._get_output(X)

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
model = FM(linear_feature_columns,dnn_feature_columns,task='binary',dnn_dropout=0.5,l2_reg_dnn=1e-2)
adam = Adam(lr=5e-6,epsilon=1e-10)

model.compile(optimizer="adam",loss = losses.binary_crossentropy,metrics=[auc,'acc'])

history = model.fit(train_model_input, train[target].values,
                    batch_size=512, epochs=50, verbose=2, validation_split=0.1)

score = model.evaluate(valid_model_input,valid[target].values)
print(model.metrics_names)
print('valid score = ' ,score)

end = time.time()
print("time:" , end - start, 's') 