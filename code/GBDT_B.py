
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import thinkstats2
import math
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from scipy.stats import entropy
from scipy.stats import ks_2samp
from sklearn import metrics
from eda_kit import conditional_entropy,roc_auc_score,auc


import lightgbm as lgb
A_train = pd.read_csv('../data/A_train.csv')
B_train = pd.read_csv('../data/B_train.csv')
B_test = pd.read_csv('../data/B_test.csv')

A_train['label'] = -1
B_train['label'] = 0
B_test['label'] = 1
B_test['flag'] = np.nan

all_data = A_train.append(B_train)
all_data = all_data.append(B_test)
all_data = all_data.reset_index(drop=True)

user_infos = [i for i in all_data.columns if 'UserInfo' in i]
product_infos = [i for i in all_data.columns if 'ProductInfo' in i]
web_infos = [i for i in all_data.columns if 'WebInfo' in i]

all_data = all_data.fillna(10)

temp_data = all_data

drop_cols_l = ['flag', 'label', 'no']
train_x = temp_data[temp_data.label==0].drop(drop_cols_l, axis=1)
train_y = temp_data[temp_data.label==0]['flag']




def lgb_feature_selection(tr_x, tr_y, model_seed =666,num_rounds = 500):
    lgb_tr = lgb.Dataset(tr_x, tr_y)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'random_state': model_seed}
    model = lgb.train(lgb_params, lgb_tr, num_boost_round=num_rounds,verbose_eval=100)
    return model




f_model = lgb_feature_selection(train_x.values, train_y.values)
lgb.plot_importance(f_model, figsize=(16,8))
features_names_im =pd.DataFrame({'feature_name':train_x.columns, 'f_value': f_model.feature_importance()})
features_used = features_names_im[features_names_im.f_value>=0.1*features_names_im.f_value.mean()]



tr_x = all_data.loc[all_data.label==0,features_used.feature_name.values].values
test_x = all_data.loc[all_data.label==1,features_used.feature_name.values].values
tr_y = all_data.loc[all_data.label==0, 'flag'].values
test_y = all_data[all_data.label==1][['no']]



from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=400, random_state=666)
gbdt.fit(tr_x, tr_y)
gbdt_pred = gbdt.predict_proba(test_x)
gbdt_pred[:, 1]
test_y['pred'] = gbdt_pred[:,1]
GBDT_B = test_y