
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


A_train = pd.read_csv('../data/A_train.csv')
B_train = pd.read_csv('../data/B_train.csv')
B_test = pd.read_csv('../data/B_test.csv')  # //
NO = B_test['no']  # //

B_train_columns = B_train.columns
B_null_count_less = []
B_null_count_large = []

# threshold = 0.63
for i in B_train_columns:
    if ((B_train[i].isnull().sum()) / len(B_train[i]) <= 0.63):
        B_null_count_less.append([i, (B_train[i].isnull().sum()) / len(B_train[i])])
    else:
        B_null_count_large.append([i, (B_train[i].isnull().sum()) / len(B_train[i])])

# len(B_null_count_less) 327

# len(B_null_count_large) 164


B_test_columns = B_test.columns
B_test_count_less = []
B_test_count_large = []

for i in B_test_columns:
    if ((B_test[i].isnull().sum()) / len(B_test[i]) <= 0.63):
        B_test_count_less.append([i, (B_test[i].isnull().sum()) / len(B_test[i])])
    else:
        B_test_count_large.append([i, (B_test[i].isnull().sum()) / len(B_test[i])])

A_feature = pd.DataFrame(B_null_count_less).values[:, 0]
B_feature = pd.DataFrame(B_null_count_less).values[:, 0]
BT_feature = pd.DataFrame(B_test_count_less).values[:, 0]

a_data = A_train[A_feature]
b_data = B_train[B_feature]
bt_data = B_train[BT_feature]

a_columns = a_data.columns
a_columns = a_columns.sort_values()  # 缺失量排序

b_columns = b_data.columns  ## B_train columns，多了一个flag
b_columns = b_columns.sort_values()

bt_columns = bt_data.columns  ## B_test columns
bt_columns = bt_columns.sort_values()

a_data = A_train[a_columns]
b_data = B_train[b_columns]
bt_data = B_test[bt_columns]

b_target = b_data['flag']
a_target = a_data['flag']

b_data.drop('flag', axis=1, inplace=True)
a_data.drop('flag', axis=1, inplace=True)

aa_data = a_data.fillna(1)
bb_data = b_data.fillna(1)
bt_data = bt_data.fillna(1)

bb_data.drop('no', axis=1, inplace=True)
aa_data.drop('no', axis=1, inplace=True)
bt_data.drop('no', axis=1, inplace=True)

import lightgbm as lgb

lgb_train = lgb.Dataset(aa_data, label=a_target)
lgb_vd = lgb.Dataset(bb_data, label=b_target)
# lgb_test = lgb.Dataset(D_feature)
# lgb_vd = lgb.Dataset(vd_x, vd_y, reference=lgb_tr)
lgb_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    #     'max_depth':18,
    #     'feature_fraction':0.85,
    #     'lambda_l1':1.2,
    'random_state': 0}  # 18     0.85

lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=200, verbose_eval=True)

lgb_pred = lgb_model.predict(bb_data)

roc_auc_score(b_target, lgb_pred)

lgb_importance = pd.DataFrame(lgb_model.feature_importance(importance_type="split"))
lgb_importance.columns = {'importance'}
columns = pd.DataFrame(b_columns).iloc[:-2, :]
columns.columns = {'feature'}
lgb_importance = pd.concat([columns, lgb_importance], axis=1)
lgb_importance = lgb_importance.sort_values(by='importance')
lgb_importance = lgb_importance[lgb_importance['importance'] > 29].reset_index().drop('index', axis=1)
lgb_importance_columns = lgb_importance['feature'].values

C_feature = A_train[lgb_importance_columns]
D_feature = B_train[lgb_importance_columns]
E_feature = B_test[lgb_importance_columns]

C_82 = C_feature['UserInfo_82']
C_82 = pd.DataFrame(C_82.fillna(C_feature['UserInfo_82'].median()))
C_82.columns = {'new_82'}

D_82 = D_feature['UserInfo_82']
D_82 = pd.DataFrame(D_82.fillna(E_feature['UserInfo_82'].median()))  # 用B——test的中位数代替有一点提升
D_82.columns = {'new_82'}

E_82 = E_feature['UserInfo_82']
E_82 = pd.DataFrame(E_82.fillna(E_feature['UserInfo_82'].median()))
E_82.columns = {'new_82'}

newC_feature = pd.DataFrame(C_feature['UserInfo_253'] * C_feature['UserInfo_242'])
newC_feature.columns = {'new_feature_1'}

newD_feature = pd.DataFrame(D_feature['UserInfo_253'] * D_feature['UserInfo_242'])
newD_feature.columns = {'new_feature_1'}

newE_feature = pd.DataFrame(E_feature['UserInfo_253'] * E_feature['UserInfo_242'])
newE_feature.columns = {'new_feature_1'}

C_feature = pd.concat([C_feature, newC_feature], axis=1)
D_feature = pd.concat([D_feature, newD_feature], axis=1)
E_feature = pd.concat([E_feature, newE_feature], axis=1)

C_feature.drop('UserInfo_134', axis=1, inplace=True)
D_feature.drop('UserInfo_134', axis=1, inplace=True)
E_feature.drop('UserInfo_134', axis=1, inplace=True)

C_feature = C_feature.fillna(1)
D_feature = D_feature.fillna(1)
E_feature = E_feature.fillna(1)

columns = D_feature.columns

lgb_params_new = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 17,
    'feature_fraction': 0.80,
    'lambda_l1': 0.6,
    #     'scale_pos_weight':1.1,
    'random_state': 0}
# 17 0.80 0.6 1550 0.591  drop134  new82       253X242    1162
lgb_train = lgb.Dataset(C_feature, label=a_target)  # 17 0.80 0.6 1550 0.589  drop134  new82            1550
lgb_vd = lgb.Dataset(D_feature, label=b_target)  # 17 0.80 0.6 1469 0.588 drop134
lgb_model = lgb.train(lgb_params_new, lgb_train, num_boost_round=2000, verbose_eval=True, valid_sets=lgb_vd,
                      early_stopping_rounds=500)  # 1256

preds = lgb_model.predict(E_feature)

A_B_LGB = []
A_B_LGB = pd.DataFrame(A_B_LGB)
A_B_LGB['no'] = no
A_B_LGB['pred'] = preds