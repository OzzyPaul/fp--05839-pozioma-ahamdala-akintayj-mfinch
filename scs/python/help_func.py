import os 
from fnmatch import fnmatch
import pandas as pd
import numpy as np
import streamlit as st

##
# 
##
def find_data(pattern= "*.csv", root = os.getcwd()):
    filenames = list()

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                filenames.append(os.path.join(path,name))
    
    if not filenames:
        print("No files found")
        return 
    else:
        return filenames

##
#
##
@st.cache
def read_file(filename, pattern= "*.csv", root = os.getcwd()):

    for path in find_data(pattern= "*.csv", root = os.getcwd()):
        if filename in path:
            return pd.read_csv(path) 
    return print("no file found")

##
# DataFrame of a given University
##

def university_df(df, univ):
    return df.loc[df['name'] == univ]

##
# MR Func
##

def MR_df(dfu):
    
    Mobility_rate = np.zeros([5,5])

    for p in range(0,5):
        for k in range(0,5):
            cond_str = "kq{}_cond_parq{}".format(k+1,p+1)
            par_str = "par_q{}".format(p+1)
            MR_temp = (dfu[par_str]*dfu[cond_str])
            Mobility_rate[p,k] = MR_temp
            
    return Mobility_rate

##
# Math Functions
##

def Mag_dis(MR):
    p_p = np.sum(MR,axis = 0)
    p_k = np.sum(MR, axis = 1)
    return p_p, p_k

def Correlation(MR):
    s = MR.shape
    p = np.array(list(range(s[1]))) + 1   
    k = np.array(list(range(s[0]))) + 1 
    p_p, p_k = Mag_dis(MR)

    up = p@p_p
    up2 = (p*p)@p_p
    uk = k@p_k
    uk2 = (k*k)@p_k
    upk = ((k[:,None]@(p[:,None]).T)*MR).sum()

    sk2 = uk2 - uk*uk
    sp2 = up2 - up*up
    skp = upk - up*uk
    rkp = skp/np.sqrt(sk2*sp2)
   
    return np.array([[sk2, skp],[skp, sp2]]), np.array([[1, rkp],[rkp, 1]])

def Norm_Mutual_Info(MR):
    
    p_p, p_k = Mag_dis(MR)

    Hp = -1*p_p*np.log2(p_p) 
    Hk = -1*p_k*np.log2(p_k)
    Hpk = -1*MR*np.log2(MR)
    I = Hk.sum() + Hp.sum() - Hpk.sum() 
    return (2*I)/(Hp.sum()+Hk.sum()) 

def Jaccard_dis(MR):
    I = Mutual_info(MR)
    Hpk = -1*MR*np.log2(MR)
    
    return  1 - I/Hpk.sum()
    
def K_L_div(MR):
    p_p, p_k = Mag_dis(MR)

    Dkl = p_p*np.log2(p_p/p_k)

    return Dkl.sum()

def skewness(MR):
    p_p, p_k = Mag_dis(MR)
    T1 = 0.5*(p_p[-1]+p_p[0]) 
    T2 = (p_p[-1]-p_p[0])
    T3 = 0.5*(p_p[-2]+p_p[1]) 
    T4 = (p_p[-2]-p_p[1])
    
    return (1/3)*(list(range(1,6))*p_p).sum()-1
