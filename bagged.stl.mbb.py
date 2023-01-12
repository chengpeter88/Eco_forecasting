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
plt.style.use('ggplot')

m3 = [8000,5120,4720,7020,3840,7360,5040,10800,6580,8520,9460,7560,7060,4560,4660,6040, 4260,5840,5020,
       10040,10060,9100,5840,6660,5680,5820, 3280, 3000,3800,6040,5040,7100,8140,6820,9000,6840,5020,	
       3020,6960,3980,3660,2740,4760,8520,4220,7160,7040,4640,7440,3900,6800,4840,3080,5220,3840,2860,	
       6780,5480,3700,4460,4180,4520,3900,3580,2120,4560,4500,5660,4240,10160,7500,7180,7800,5240,5480,	
       3900,2920,4100,4540,	2520,4920	
    ]
m3_full=[8000,5120,4720,7020,3840,7360,5040,10800,6580,8520,9460,7560,7060,4560,4660,6040, 4260,5840,5020,
       10040,10060,9100,5840,6660,5680,5820, 3280, 3000,3800,6040,5040,7100,8140,6820,9000,6840,5020,
       3020,6960,3980,3660,2740,4760,8520,4220,7160,7040,4640,7440,3900,6800,4840,3080,5220,3840,2860,
       6780,5480,3700,4460,4180,4520,3900,3580,2120,4560,4500,5660,4240,10160,7500,7180,7800,5240,5480,
       3900,2920,4100,4540,	2520,4920,3080,2800,2160,3500,3420,3480,1900,2220,2620,5160,2520,4740,3380,3380,
       2640,3640,4840,1700,2660,2920,1860,3140,3360,4400,4760,5980,3980,4040,4440,1340,1740,2420,4380,4600,3800,
       3020,4920,4040,2960,4220,1660,2280,1600,2940,2960]

###一維度 dataframe tX1
m3= pd.Series(
    m3, index=pd.date_range("10-1-1984", periods=len(m3), freq="M"), name="m3"
    )
m3_full= pd.Series(
    m3_full, index=pd.date_range("1984-10-1", periods=len(m3_full), freq="M"), name="m3_full")

m3.describe()
T=len(m3)
m3.plot()
plt.show(block=True)
#print(m3)
m3_full.plot()
plt.show(block=True)
#print(m3_full)


stl=STL(m3,seasonal=13)
res=stl.fit()
trend = res.trend
seasonal = res.seasonal
res.plot()
#dir(res)

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
z=pd.Series(z,index=m3.index)
z.plot()
plt.show(block=True)

#  moving block bootstrap
l = 24 # block size
B = 11 # number of bootstrapped series
T1=len(m3)+len(range(1,13))
fcast_h=list(range(1,13))

bt_m3 = pd.DataFrame(np.zeros((len(m3),B)),index=m3.index) #
m3_fcast=pd.DataFrame(np.zeros((T1,B)),index=pd.date_range(start=m3.index[0],periods=T1,freq="M")) 

for bb in range(B):
    z = mbb(res.resid,l) 
    bt_m3.iloc[:,bb]= np.array(z)+trend+seasonal
    m3_fcast.iloc[len(m3):,bb]=AutoETS(auto=True,n_jobs=-1,sp=12,maxiter=5000).fit_predict(bt_m3.iloc[:,bb],fh=fcast_h)
    print(m3_fcast.iloc[len(m3):,bb])
    print(AutoETS(auto=True,n_jobs=-1,sp=12,maxiter=5000).fit(bt_m3.iloc[:,bb],fh=fcast_h).summary())
    m3_fcast.iloc[:len(m3),bb]=m3
####跑11條mbb之下最小的aic ETSmodel選到ＭＡＭ aic=1425.585
m3_fcast["1991":].plot(legend=False,figsize=(10,8),title="m3_fcast",xlabel="time")
plt.show(block=True)


#單純ETS model
model = ETSModel(m3,error="mul",trend="add",seasonal="mul",damped_trend=None,seasonal_periods=12)
fit=model.fit()
print(fit.summary())
pred=fit.get_prediction(start="1991-07-31", end="1992-06-30") #12條月資料
#outsample  forecast
pre_summary=pred.summary_frame(alpha=0.05)
pre_dataframe=pd.DataFrame(pre_summary)
pre_dataframe.iloc[:,0]
pred.summary_frame(alpha=0.05).plot(figsize=(10,8))
plt.show(block=True)

# 最小的aic選到其中一條mbb forecast ETS(MNM) by bagged.STL.mbb
#actual data 從1991-07-31 往後12 period
actual   = [3080,2800,2160,3500,3420,3480,1900,2220,2620,5160,2520,4740]
forecast = [6410.713130,6609.706439,5582.294003,4953.022845,3456.171821,4988.803880,3828.434897,3061.715629,
            4313.181609,4182.465532,5783.649304,5719.322309]
APE = []
for i in range(12):
    per_err = (actual[i] - forecast[i]) / actual[i]
    per_err = abs(per_err)
    APE.append(per_err)
MAPE = sum(APE)/len(APE)
print(f'''
MAPE   : { round(MAPE, 2) }
MAPE % : { round(MAPE*100, 2) } %
''')

#單純只做ETS model 採用(MNM)
forecast_ets= [6608.335766,5959.495600,5115.023692,4939.472017,3728.716525,
               4323.364187,3714.682341,2644.331966,4093.768477,3643.747768,
               5790.937028,5201.152065]

APE_ets = []
for i in range(12):
    per_err_ets = (actual[i] - forecast_ets[i]) / actual[i]
    per_err_ets = abs(per_err_ets)
    APE_ets.append(per_err)
MAPE = sum(APE_ets)/len(APE_ets)
print(f'''
MAPE   : { round(MAPE, 2) }
MAPE % : { round(MAPE*100, 2) } %
''')








