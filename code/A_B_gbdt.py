
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

# In[67]:

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

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=2, random_state=0)  #
clf.fit(aa_data, a_target)
y_pred = clf.predict_proba(bb_data)
y_pred = pd.DataFrame(y_pred).iloc[:, 1]
roc_auc_score(b_target, y_pred)

# clf.fit(aa_data,a_target)
clf_importance = clf.feature_importances_
clf_importance_ = pd.DataFrame(clf_importance)
clf_importance_.columns = {'importance'}
bb_columns = pd.DataFrame(bb_data.columns)
bb_columns.columns = {'feature'}

# 影响度排序
clf_feature_values = pd.concat([bb_columns, clf_importance_], axis=1)
# feature_values.columns = {'importance','feature'}
clf_feature_values = clf_feature_values.sort_values(by='importance')

# 影响度非0的特征
clf_feature_well = clf_feature_values[clf_feature_values['importance'] != 0]
clf_feature_well_columns = clf_feature_well['feature'].values
clf_feature_well.index = clf_feature_well_columns

columns_GBDT = clf_feature_well.index

C_feature = A_train[columns_GBDT]
D_feature = B_train[columns_GBDT]
E_feature = B_test[columns_GBDT]

C_flag = pd.DataFrame(A_train['flag'])
D_flag = pd.DataFrame(B_train['flag'])

C_82 = C_feature['UserInfo_82']
C_82 = pd.DataFrame(C_82.fillna(C_feature['UserInfo_82'].median()))
C_82.columns = {'new_82'}

D_82 = D_feature['UserInfo_82']
D_82 = pd.DataFrame(D_82.fillna(E_feature['UserInfo_82'].median()))  # 用B——test的中位数代替有一点提升
D_82.columns = {'new_82'}

E_82 = E_feature['UserInfo_82']
E_82 = pd.DataFrame(E_82.fillna(E_feature['UserInfo_82'].median()))
E_82.columns = {'new_82'}

newC_feature = pd.DataFrame(C_feature['UserInfo_82'] * C_feature['UserInfo_222'])
newC_feature.columns = {'new_feature_1'}

newD_feature = pd.DataFrame(D_feature['UserInfo_82'] * D_feature['UserInfo_222'])
newD_feature.columns = {'new_feature_1'}

newE_feature = pd.DataFrame(E_feature['UserInfo_82'] * E_feature['UserInfo_222'])
newE_feature.columns = {'new_feature_1'}

C_feature = pd.concat([C_feature, C_82], axis=1)
D_feature = pd.concat([D_feature, D_82], axis=1)
E_feature = pd.concat([E_feature, E_82], axis=1)

C_feature = pd.concat([C_feature, newC_feature], axis=1)
D_feature = pd.concat([D_feature, newD_feature], axis=1)
E_feature = pd.concat([E_feature, newE_feature], axis=1)

C_feature = C_feature.fillna(1)
D_feature = D_feature.fillna(1)
E_feature = E_feature.fillna(1)

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=149, learning_rate=0.66, max_depth=2, random_state=0, max_features=14,
                                 min_weight_fraction_leaf=0.11)
clf.fit(C_feature, C_flag)

y_pred = clf.predict_proba(D_feature)
y_pred = pd.DataFrame(y_pred).iloc[:, 1]
roc_auc_score(D_flag, y_pred)

y_pred = clf.predict_proba(E_feature)
y_pred = pd.DataFrame(y_pred).iloc[:, 1]
b = pd.DataFrame(y_pred)

no = pd.DataFrame(NO)

A_B_GBDT = []
A_B_GBDT = pd.DataFrame(A_B_GBDT)
A_B_GBDT['no'] = no
A_B_GBDT['pred'] = y_pred