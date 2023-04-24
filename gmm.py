###library
from numpy import hstack, ones, array, mat, tile, reshape, squeeze, eye, asmatrix
from numpy.linalg import inv
from pandas import read_csv, Series 
from scipy.linalg import kron
from scipy.optimize import fmin_bfgs
import numpy as np
import statsmodels.api as sm
import pandas as pd

iteration=0
lastvalue=0
functionCount=0
###
def iter_print(params):
    global iteration ,lastvalue, functionCount
    iteration +=1
    print("Func value:{0:},Iteration:{1:},Function Count{2:}".format(lastvalue, iteration, functionCount))

###
def GMM_objective(params,pRets,fRets,Winv,out=False):
    global lastvalue , functionCount
    T,N=pRets.shape
    T,K=fRets.shape
    beta=squeeze(array(params[:(N*K)])) ### from 0 to N*K(former)
    lam=squeeze(array(params[(N*K):]))   ###squeeze compress the dimension to the vector
    beta=reshape(beta,(N,K))
    lam=reshape(lam,(K,1))
    betalam=beta @ lam  ###@  martix Multiplication
    expectRet=fRets @ beta.T
    e=pRets- expectRet
    instr=tile(fRets,N)
    moment1=kron(e,ones(1,K))  ###inner product
    moment1=moment1*instr
    moment2=pRets-betalam.T
    moments=hstack((moment1,moment2))  ###水平方向进行叠加
    avgMoment=moments.means(axis=0)
    J=T*mat(avgMoment)*mat(Winv)*mat(avgMoment).T ###object function
    J=J[0,0]
    lastvalue =J
    functionCount +=1
    if not out:
        return J
    else:
        return J, moments

###
def gmm_G(params,pRets,fRets):
    T,N=pRets.shape
    T,K=fRets.shape
    beta=squeeze(array(params[:(N*K)]))
    lam=squeeze(array(params[(N*K):]))
    beta=reshape(beta,(N,K))
    lam=reshape(lam,(K,1))
    G=np.zeros(N*K+K,N*K+K)
    ffp=(fRets.T @ fRets)/T
    G[:(N*K),:(N*K)]=kron(eye(N),ffp)
    G[:(N*K),(N*K):] = kron(eye(N),-lam)
    G[(N*K):,(N*K):] = -beta.T
    return G


###
data=pd.read_csv("FamaFrench.csv")
dates = data['date'].values
factors = data[['VWMe','SMB','HML']].values
riskfree = data['RF'].values
portfolios = data.iloc[:,5:].values

T,N = portfolios.shape
portfolios = portfolios[:,np.arange(0,N,2)]
T,N = portfolios.shape
excessRet = portfolios - np.reshape(riskfree,(T,1))
K = np.size(factors,1)


betas = []
for i in range(N):
    res = sm.OLS(excessRet[:,i],sm.add_constant(factors)).fit()
    betas.append(res.params[1:])

avgReturn = excessRet.mean(axis=0)
avgReturn.shape = N,1
betas = array(betas)
res = sm.OLS(avgReturn, betas).fit()
riskPremia = res.params



riskPremia = res.params
riskPremia