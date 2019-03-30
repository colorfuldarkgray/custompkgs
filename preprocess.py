# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import pandas as pd
class N_tt_tn:

    def __init__(self,ttp,fixed,pnb): # ttp es test percentage 
        self.ttp=ttp
        self.fixed=fixed
        self.pnb=pnb
        self.ttsz=int(np.round(self.ttp*len(self.fixed)))
        self.rows,self.cols=self.fixed.shape
        self.tenrandrow=np.zeros((self.pnb,self.ttsz)) # pnb partition number
        self.test=[] # lista de pnb combinaciones para pruebas
        self.train=[] # lista de pnb combinaciones para entrenamiento
        def getrandrows(self):
            self.randrow=np.random.randint(0,self.rows,2*self.ttsz) 
            self.randrow=np.unique(self.randrow)
            self.randrow=self.randrow[:self.ttsz] # 
            return self.randrow
        for i in range(pnb):
            self.tenrandrow[i,:]=getrandrows(self)
            self.test.append(self.fixed.iloc[self.tenrandrow[i],:].copy())
            self.train.append(self.fixed.drop(self.tenrandrow[i]).copy())
            # print(train[i].shape, test[i].shape)

    
    
