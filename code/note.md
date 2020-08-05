#环境配置
* 安装anaconda
* conda 环境，运行 `conda create -n qhzx37 python=3.7; `
* 运行`conda install -n qhzx37 numpy pandas scikit-learn scipy seaborn matplotlib py-xgboost lightgbm ipython jupyter`
* pycharm配置：
    * 添加qhzx37作为interpreter
    * 标记 代码`code`目录作为source directory
    * 在Python Console里面增加启动命令，方便在console里面调试
```python
      
import sys
import os
os.chdir('./code')
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
```

#EDA
* eda 部分在eda.ipynb
* eda 里面的代码部分来源于： https://www.kesci.com/home/project/59e74d916d213335f38d7f72 的XGB_B
* XGB_B.ipynb GBDT_B.py A_B_gbdt.py A_B_lgb.py来源于： https://www.kesci.com/home/project/59e74d916d213335f38d7f72