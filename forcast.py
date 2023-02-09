import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox 
from scipy.special import inv_boxcox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS 
from itertools import product
import warnings 
from scipy.special import inv_boxcox
plt.style.use('ggplot')

m3_full=[8000,5120,4720,7020,3840,7360,5040,10800,6580,8520,9460,7560,7060,4560,4660,6040, 4260,5840,5020,
       10040,10060,9100,5840,6660,5680,5820, 3280, 3000,3800,6040,5040,7100,8140,6820,9000,6840,5020,
       3020,6960,3980,3660,2740,4760,8520,4220,7160,7040,4640,7440,3900,6800,4840,3080,5220,3840,2860,
       6780,5480,3700,4460,4180,4520,3900,3580,2120,4560,4500,5660,4240,10160,7500,7180,7800,5240,5480,
       3900,2920,4100,4540,	2520,4920,3080,2800,2160,3500,3420,3480,1900,2220,2620,5160,2520,4740,3380,3380,
       2640,3640,4840,1700,2660,2920,1860,3140,3360,4400,4760,5980,3980,4040,4440,1340,1740,2420,4380,4600,3800,
       3020,4920,4040,2960,4220,1660,2280,1600,2940,2960]

###一維度 dataframe

m3_full= pd.Series(
    m3_full, index=pd.date_range("1984-10-1", periods=len(m3_full), freq="M"), name="m3_full")

T=len(m3_full)
#m3_full.plot()
#plt.show(block=True)


m3_full_transformed, lambda_ = boxcox(m3_full)

stl = STL(m3_full_transformed, seasonal=13,period=12)
res=stl.fit()
trend = res.trend
seasonal = res.seasonal
plt.rcParams["figure.figsize"]=(30,15)
res.plot()
def mbb(x,l):  #l block size
    n=len(x) #the length of data
    nb=np.int(n/l)+2
    idx=np.random.randint(n-l,size=nb)
    z=[]
    for ii in idx:
        z=z+list(x[ii:ii+l]) 
    z=z[np.random.randint(l):]
    z=z[:n]
    return(z)

z =mbb(res.resid,8)
z=pd.Series(z,index=m3_full.index)
#z.plot()
#plt.show()

#  moving block bootstrap
l = 24 # block size
B = 11 # number of bootstrapped series
T1=len(m3_full)+len(range(1,13))
fcast_h=list(range(1,13))

bt_m3_full = pd.DataFrame(np.zeros((len(m3_full),B)),index=m3_full.index) #
m3_full_fcast=pd.DataFrame(np.zeros((T1,B)),index=pd.date_range(start=m3_full.index[0],periods=T1,freq="M")) 

for bb in range(B):
    z = mbb(res.resid,l) 
    bt_m3_full.iloc[:,bb]= np.array(z)+trend+seasonal

inv_box_cox =inv_boxcox(bt_m3_full, lambda_)
inv_box_cox 

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
MAPEs = []
for bb in range(B):
    inv_box_cox_series = inv_box_cox.iloc[:,bb]
    X_train, X_test = inv_box_cox_series[:-12], inv_box_cox_series[-12:]
    autoets = AutoETS(auto=True,n_jobs=-1,sp=12,maxiter=5000)
    autoets.fit(X_train)
    #print(autoets.summary())
    y_pred = autoets.predict(fcast_h)
    mape = mean_absolute_percentage_error(X_test, y_pred)
    print("this is mape ",mape)
    MAPEs.append(mape)
    plt.subplot(B, 1, bb + 1)
    plt.plot(X_train, label="data_train")
    plt.plot(X_test,label="data_test")
    plt.plot(X_test.index, y_pred, label="forecast")
    plt.legend()
best_model = MAPEs.index(min(MAPEs))
print("model no",best_model)
plt.show()



