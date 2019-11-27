# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:32:09 2019

@author: xorez

Erick Rico.
Diseño y Análisis de Algoritmos.
CIC IPN
"""
import numpy as np
from sls_class import sls

s = sls('test_data.txt', isRandom=False, rangeRandom=0)  #Load the test data set or generate by itself

s.findOpt(penaltyFactor=0.3)

n = len(s.x)
minX = np.amin(s.x)
maxX = np.amax(s.x)
dX = 0.33*(maxX-minX)/(n-1)
xPlt = np.arange(minX,maxX+dX,dX)

s.plotFit(xPlt)