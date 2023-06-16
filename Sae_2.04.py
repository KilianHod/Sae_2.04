import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
import warnings

#Etape 1
SaeDF=pd.read_csv("VueSae_2.04.csv")

#Etape 2
SaeDF = SaeDF.dropna()
SaeArray=SaeDF.to_numpy()
                                            
#Etape 3
def CentreReduire (T):
    T = np.array(T, dtype=np.float64)
    TMoy=np.mean(T, axis=0)
    TEcart=np.std(T, axis=0)
    (n,p)=T.shape
    res=np.zeros((n,p))
    for j in range(p):
        res[:,j]=(T[:,j]-TMoy[j])/TEcart[j]
    return res

SaeArrayCR = CentreReduire(SaeArray[:,1:])
    
#PARTIE 2
#Etape 1 
MatriceCov = np.cov(SaeArrayCR, rowvar=False)

#Etape 2
Y = SaeArrayCR [:,0]
X = SaeArrayCR[:,[2,3,4]]

linear_regression = LinearRegression()
linear_regression.fit(X,Y)

a = linear_regression.coef_

#Etape 3
CorSae = linear_regression.score(X,Y)