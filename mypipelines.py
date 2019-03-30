# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:43:44 2019

SE RECICLAN LAS VARIABLES PARA EVITAR ERROR DE MEMORIA

@author: Rodrigo
"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
def polyft_lrr(grado,train,test):
    polyft=(PolynomialFeatures(degree=grado))
    Xtnpf=[]
    Xttpf=[]
    for i in range(len(train)): # 10 particiones
        Xtnpf.append(polyft.fit_transform(train[i].loc[:,'X0':'X12']))
        Xttpf.append(polyft.fit_transform(test[i].loc[:,'X0':'X12']))

    ppredtst=[] #predicciones sobre datos validación
    ppredtrn=[] #predicciones sobre datos entrenamiento
    plrm=[] # modelos de regresion lineal
    error_emp=np.zeros((1,2))

    for i in range(len(train)):
        plrm.append(linear_model.LinearRegression())
        plrm[i].fit(Xtnpf[i],train[i].loc[:,'Y']) 
        ppredtst.append(plrm[i].predict(Xttpf[i]))
        ppredtrn.append(plrm[i].predict(Xtnpf[i]))
        error_emp[0,0]+=(mean_squared_error(test[i].Y,ppredtst[i]))
        error_emp[0,1]+=(mean_squared_error(train[i].Y,ppredtrn[i]))
    error_emp[0,0]/=len(train)
    error_emp[0,1]/=len(train)
    return error_emp