import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import os as os 
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

plt.close("all")
def isfloat(n):
    try: 
        float(n)
        return True
    except ValueError:
        return False

def data(A1):
    F=[]
    for i in A1:
        if isfloat(i) is True:
            F.append(i)
        else: 
            break
    return np.array(F)
    
def std(a,b,c):
    mean=(a+b+c)/3
    std=np.sqrt(((a-mean)**2+(b-mean)**2+(c-mean)**2)/2)
    return std

def Get_data(file,A1,A2,A3):
    datos=pd.read_excel(file)
    c1=datos[A1]
    j1=data(c1) 
    c2=datos[A2]
    j2=data(c2) 
    c3=datos[A3]
    j3=data(c3)   
    st=[]
    mean=[]
    for i in range(0,len(j1)):
        mean.append((j1[i]+j2[i]+j3[i])/3)
        st.append(std(j1[i],j2[i],j3[i]))
        
    return np.array(mean),np.array(st)

#FORWARD
def dyf(a,c): # a es un arreglo de los valores de la variable dependiente, c es un arreglo de los valores de la variable independiente 
    dyforward=[0.0]*len(a)
    for i in range(len(a)-1):
        dyforward[i]=(c[i+1]-c[i])/(a[i+1]-a[i])
        dyforward[-1]=((c[-1]-c[-2])/(a[-1]-a[-2]))
    return dyforward

#BACKWARD
def dyb(a,c):
    dyback=[0.0]*len(a)
    dyback[0]=0
    for i in range(len(a)):
        dyback[i]=(c[i]-c[i-1])/(a[i]-a[i-1])   
    return dyback     
     
#CENTRAL
def dyc(a,c):
    dycen=[0.0]*len(a)
    dycen[0]=((c[0]-c[1])/(a[0]-a[1]))
    for i in range(len(a)-1):
        dycen[i]=(c[i+1]-c[i-1])/(a[i]-a[i-1])               
        dycen[-1]=(c[-1]-c[-2])/(a[-1]-a[-2])  
    return np.array(dycen)   

def NewData(x,n): 
    H=[]
    for i in range(n,len(x)):
        H.append(x[i])
    return np.array(H)
    
def FAMconc(a,m,b): return (a-b)/m

def Mean_Slope(x,n): 
    m=[]
    for i in range(0,n):
        m.append(x[i])
    m=np.array(m)
    mean=stats.mean(m)
    std=stats.stdev(m)
    return mean,std
def lineal(x,a,b): return x*a+b

def Opt_lineal(x,y):
    min_r=0.99
    r_sqrt=0
    while r_sqrt < min_r:
        x=x[:-1]
        y=y[:-1]
        param,covM=np.polyfit(x,y,1, cov=True)
        r_sqrt=np.corrcoef(x,y)[0,1]**2
        s=np.array(abs(((np.sqrt(np.diag(covM))/1)))).tolist()
    return param, s, r_sqrt

def CF_C(x,std,m,b):
    stdf=(std/x)
    conc=FAMconc(x,m,b)
    stdfi=2*stdf*conc
    return conc,stdfi
