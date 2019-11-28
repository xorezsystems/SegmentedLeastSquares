# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 12:06:47 2019

@author: xorez

Erick Rico.
Diseño y Análisis de Algoritmos.
CIC IPN

"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import os

class sls:
    
    def __init__(self, filename, isRandom, rangeRandom):
        if isRandom:
            try:
                self.generateData(rangeRandom)
            except Exception:
                 print("An error has ocurred")
            finally:
                self.getData('randomData'+str(rangeRandom)+'.txt')
        else:
            self.getData(filename) 
        
      
    def generateData(self, rangeRandom):
        dataX = []
        dataY = []
        for i in range (rangeRandom):
            dataX.append(randint(1, 100))
            dataY.append(randint(1, 100))
        
        file = open('randomData'+str(rangeRandom)+'.txt', "w")
        for i in range (len(dataX)):
            file.write(str(dataX[i])+ ' '+str(dataY[i])+ os.linesep)
            
        file.close
        
    def getData(self, filename):
        with open(filename) as csvfile:
            raw = csv.reader(csvfile, delimiter = ' ')
            data = np.array([[float(row[0]), float(row[1])] for row in raw])
            
            data=np.asarray(sorted(data, key=lambda x:x[0]))
            print(sorted(data, key=lambda x:x[0]))
            
            
        xCordenade = data[:,0]
        yCordenade  = data[:,1]
        variance = np.std(yCordenade)**2   #variance of "y" for caculate penalty
        n = len(xCordenade)
        xVal=0
        b=0
        errArr = np.zeros((n,n))
        aArr = np.zeros((n,n))
        bArr = np.zeros((n,n))
        
        self.calcLeastSquares(xCordenade[0:n+1],yCordenade[0:n+1]) #Calling calcLeastSquares for getting a line comparison
        
        
        for j in range(n): #for j  end indices of data, 0 to n-1
            for i in range(j+1):  #for i starting indices of this data, 0 to j
                a, b, err2 = self.lsCoef(xCordenade[i:j+1],yCordenade[i:j+1]) #for this segment, i to j, get coefs and error of fit
                
                
                errArr[i,j] = err2 #store error
                aArr[i,j] = a #store coefs
                bArr[i,j] = b
        
        self.x = xCordenade
        self.y = yCordenade
        self.errArr = errArr
        self.aArr = aArr 
        self.bArr = bArr
        self.variance = variance
        
    def calcLeastSquares(self, x, y):
        print("..:: Mínimos Cuadrados ::.. ")
        n=len(x)
        if (n == 1):
            return (0.0, y[0], 0.0)
        
        sumX = np.sum(x)
        print("x: ", x)
        sumY = np.sum(y)
        sumX2 = np.sum(x ** 2)
        sumXY = np.sum(x * y)
        m = (n * sumXY - sumX * sumY)/(n * sumX2 - abs(sumX * sumX))  
        self.b = (sumY * sumX2 - sumX * sumXY) / (n * sumX2 - abs(sumX * sumX))
        self.xVal = -self.b / m
        print("sx: ", sumX)
        print("sy: ", sumY)
        print("sx2: ", sumX2)
        print("sxy: ", sumXY) 
        print("m: ", m) 
        print("b: ", self.b) 
        print("xVal: ", self.xVal) 
        
        print("-------------------------------------") 

    def lsCoef(self, x, y):
        n=len(x)
        if (n == 1):
            return (0.0, y[0], 0.0)
        
        sumX = np.sum(x)
        sumY = np.sum(y)
        sumX2 = np.sum(x ** 2)
        sumXY = np.sum(x * y)
        a = (n*sumXY - sumX * sumY)/(n*sumX2 - sumX * sumX)   
        b = (sumY - a * sumX)/n
        err2 = np.sum((y - a * x - b)**2)

        return (a, b, err2)
        
    def findOpt(self, penaltyFactor):
        penalty = self.variance * penaltyFactor
        n = self.errArr.shape[0]
        
        #Get minimum error over possible start indices
        optArr = np.zeros(n)
        for j in range(n):
            #Use min error for previous end index and current error to get errors for all possible start indices
            tmpOpt = np.zeros(j+1)
            tmpOpt[0] = self.errArr[0,j] + penalty
            for i in range(1,j+1):
                tmpOpt[i] = optArr[i-1] + self.errArr[i,j] + penalty
            optArr[j] = np.amin(tmpOpt)
        
        #Get segment and coefficients
        optCoefs = []
        j = n-1
        while j >= 0:
            tmpOpt = np.zeros(j+1)
            tmpOpt[0] = self.errArr[0,j] + penalty
            for i in range(1,j+1):
                tmpOpt[i] = optArr[i-1] + self.errArr[i,j] + penalty
            iOpt = np.argmin(tmpOpt)
            aOpt = self.aArr[iOpt,j]
            bOpt = self.bArr[iOpt,j]
            
            #Set boundaries of interval for these coefs in terms of x
            if iOpt <= 0:
                xMin = np.NINF
            else:
                xMin = (self.x[iOpt-1] + self.x[iOpt])/2
            if j >= n-1:
                xMax = np.Inf
            else:
                xMax = (self.x[j] + self.x[j+1])/2     
            optCoefs.insert(0, (xMin,xMax,aOpt,bOpt))
            j = iOpt-1
            
        self.optCoefs = optCoefs
        
    def getFit(self,x):
        #Get fit using coefs
        n = len(x)
        yFit = np.zeros(n)
        for optCoef in self.optCoefs:
            ind = [i for i,elem in enumerate(x) \
                   if elem >= optCoef[0] and elem <= optCoef[1]]
            yFit[ind] = x[ind] * optCoef[2] + optCoef[3]

        return yFit
        
    def plotFit(self,xplt):
        yfit = self.getFit(xplt)
        plt.plot(self.x, self.y, '.')
        #plt.plot([abs(self.xVal), 0], [0, self.b], color='red', marker = '.')
        plt.plot(xplt, yfit, color='green')
        plt.show()