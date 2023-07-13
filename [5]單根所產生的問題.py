# -*- coding: utf-8 -*-
"""[5]單根所產生的問題.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sXEVnWIDm3V_lAjpvoaJu9Yfs8zTG-Gi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

"""# 單根所產生的問題
1. 估計的向下偏誤
2. 虛假回歸
3.

## 估計的向下偏誤
"""

# 生成單根過程  可生成諸多個error
T = 50
# error term 數 時間Ｔ越大越精準
sigma = 0.20
#sigma 乘上常態
yt = (np.random.randn(T)*sigma).cumsum()
#randn 生成標準常態 乘上sigma成為白噪音的模型 期望 0 變異 sigma＊2 /cumsum為殘差的累加
plt.plot(yt)

yt = pd.Series(yt)  #成為序列 可以做平移
yt.shift()

# 估計AR(1)  yt=beta0 +beta1 yt-1 +error
y = yt
x = sm.add_constant(yt.shift())
result = sm.OLS(y,x,missing='drop').fit()  #回歸
print(result.summary())   #p value <0.05
#存在偏誤的發生
#beta1 coef 一定小於1

beta = result.params[0]  #抓取0變數

#bias = abs(1-beta)
print(bias)

# 記錄不同T所產生的誤差bias

xx=pd.Series(np.random.randn(100).cumsum())
yy=pd.Series(np.random.randn(100).cumsum())

y=yy
x=sm.add_constant(xx)
result= sm.OLS(y,x,missing='drop').fit()
print(result.summary())
#p value <0.05 顯著

result.pvalues[0]<0.05



