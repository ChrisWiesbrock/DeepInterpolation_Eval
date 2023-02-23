# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:15:13 2023

@author: wiesbrock
"""

from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import uniform_filter1d

path=r'C:\Users\wiesbrock\Desktop\Deepinter Eval\*.csv'
file_list=glob.glob(path)

ending=[]

for i in range(len(file_list)):
    if file_list[i][-8:]=='Deep.csv':
        deep_trace_path=file_list[i]
    else:
        raw_trace_path=file_list[i]
        
raw_df=pd.read_csv(raw_trace_path)
deep_df=pd.read_csv(deep_trace_path)

header = list(raw_df.columns)
header=header[1:]
b=[]
m=[]
mean_raw=[]
std_raw=[]
mean_deep=[]
std_deep=[]
r_list=[]

for i in header:
    plt.figure()
    trace_raw=raw_df[i]
    trace_deep=deep_df[i]
    trace_raw=np.array(trace_raw)
    trace_deep=np.array(trace_deep)
    trace_deep = trace_deep[~np.isnan(trace_deep)]
    trace_raw = trace_raw[~np.isnan(trace_raw)]
    coef = np.polyfit(trace_raw, trace_deep, 1)
    m.append(coef[0])
    b.append(coef[1])
    mean_raw.append(np.mean(trace_raw))
    std_raw.append(np.std(trace_raw))
    mean_deep.append(np.mean(trace_deep))
    std_deep.append(np.std(trace_deep))
    poly1d_fn = np.poly1d(coef) 
    r=stats.pearsonr(trace_raw,trace_deep)
    r_list.append(r[0])
    x=np.linspace(np.min(trace_raw),np.max(trace_raw))
    plt.title(i+" "+str(r[0]))
    plt.plot(x, poly1d_fn(x), '--k')
    plt.ylabel('Deep')
    plt.xlabel('raw')
    plt.scatter(trace_raw,trace_deep)
    if r[0]<0.9:
        #y_1 = uniform_filter1d(trace_deep, size=len(trace_raw))
        #y= uniform_filter1d(trace_deep, size=len(trace_deep))
        plt.figure()
        plt.plot(trace_deep,'k-')
        plt.plot(trace_raw, 'g-')
        #plt.plot(y_1, 'g--')
        #plt.plot(y, 'k--')
        
mean_raw=np.array(mean_raw)
std_raw=np.array(std_raw)
mean_deep=np.array(mean_deep)
std_deep=np.array(std_deep)

diff_mean=mean_raw-mean_deep
diff_std=std_raw-std_deep
        



        
