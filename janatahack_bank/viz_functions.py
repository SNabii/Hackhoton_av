#!/usr/bin/env python
# coding: utf-8

# <p>Python Module <span style="color:blue;font-weight:bold">Importing Library</span> with the function written below <span style="color:darkolivegreen;font-weight:bold">dark green</span> eyes.</p>

# In[3]:

# basic import's
import os
import pandas as pd 
import numpy as np 

# stats
import scipy.stats as stats
from scipy.stats import kurtosis, skew

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Nabi My-Functions ---------------------------------------------------------------------------------------


def call_univariate_cat_y2label_indiv(df,cat_col,y_lab,Top_n,thres,ytag,prt):
    i = cat_col#"Age_Of_Vehicle"
    y2 = y_lab#"Renewed2"
    Top_n = Top_n #15
    ytag = ytag
    col_count  = df[f'{i}'].value_counts()
    #print(col_count)
    col_count = col_count[:Top_n,]

    col_count1 = df[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp = pd.crosstab(df[f'{i}'], df[f'{y2}'], normalize='index') * 100
    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    tmp.rename(columns={0:'NotRenwed%', 1:'Renewed%'}, inplace=True)
    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0

    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={0:'NR_count', 1:'R_count'}, inplace=True)
    if 'NR_count' not in tmp1.columns:
        print("NR_count is not present in ",i)
        tmp1['NR_count'] = 0
    if 'R_count' not in tmp1.columns:
        print("R_count is not present in ",i)
        tmp1['R_count'] = 0

    tmpz=pd.merge(tmp,tmp1,
        left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = (tmpz['R_count']/tmpz['Tot'])*100
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    #tmpz.sort_index(inplace=True)
    if prt == 'Y':
        print(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        tmpzi.to_excel("tmpz.xlsx")
      
   
    plt.figure(figsize=(16,7))
    g=sns.barplot(tmpz.index, tmpz[f'{i}'], alpha=0.8,order = col_count.index)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.title(f'{i} with {vol_inperc}%', fontsize = 16,color='blue')
    #g.set_title(f'{i}')
    plt.ylabel('Volume', fontsize=12)
    plt.xlabel(f'{i}', fontsize=12)
    plt.xticks(rotation=90)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{}\n{:1.2f}%'.format(round(height),height/len(df)*100),
            ha="center", fontsize=10, color='blue')

    gt = g.twinx()

    if ytag == 1:
        gt = sns.pointplot(x=tmpz.index, y='Renewed%', data=tmpz, color='green', legend=True,order=tmpz.index)
    if thres is np.nan:
        gt.set_ylim(0,100)
   

    if ytag == 0:
        gt = sns.pointplot(x=tmpz.index, y='NotRenwed%', data=tmpz, color='red', legend=True,order=tmpz.index)
        gt.set_ylim(tmp['NotRenwed%'].min()-1,tmp['NotRenwed%'].max()+5)
        
        
    if thres is np.nan: 
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
    j=0
    for c in gt.collections:
        for i, of in zip(range(len(c.get_offsets())), c.get_offsets()):
            gt.annotate(values[j], of, color='brown', fontsize=12, rotation=45)
            j += 1
    #plt.show()
    
    
def call_univariate_ord_y2label_indiv(df,cat_col,y_lab,Top_n,thres,ytag,prt):
    i = cat_col#"Age_Of_Vehicle"
    y2 = y_lab#"Renewed2"
    Top_n = Top_n #15
    ytag = ytag
    
    #df[f'{i}'] = df[f'{i}'].astype(int)
    col_count  = df[f'{i}'].value_counts()
    #print(col_count)
    col_count.sort_index(inplace=True)
    col_count = col_count[:Top_n,]

    col_count1 = df[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp = pd.crosstab(df[f'{i}'], df[f'{y2}'], normalize='index') * 100
    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    tmp.rename(columns={0:'NotRenwed%', 1:'Renewed%'}, inplace=True)
    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0

    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={0:'NR_count', 1:'R_count'}, inplace=True)
    if 'NR_count' not in tmp1.columns:
        print("NR_count is not present in ",i)
        tmp1['NR_count'] = 0
    if 'R_count' not in tmp1.columns:
        print("R_count is not present in ",i)
        tmp1['R_count'] = 0

    tmpz=pd.merge(tmp,tmp1,
        left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = tmpz['Renewed%'].mean()
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    #tmpz.sort_index(inplace=True)
    
    
    if prt == 'Y':
        print(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        tmpzi.to_excel("tmpz.xlsx")
      
   
    plt.figure(figsize=(16,7))
    g=sns.barplot(tmpz.index, tmpz[f'{i}'], alpha=0.8)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.title(f'{i} with {vol_inperc}%', fontsize = 16,color='blue')
    #g.set_title(f'{i}')
    plt.ylabel('Volume', fontsize=12)
    plt.xlabel(f'{i}', fontsize=12)
    plt.xticks(rotation=90)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{}\n{:1.2f}%'.format(round(height),height/len(df)*100),
            ha="center", fontsize=10, color='blue')

    gt = g.twinx()

    if ytag == 1:
        gt = sns.pointplot(x=tmpz.index, y='Renewed%', data=tmpz, color='green', legend=True)
    
    if ytag == 0:
        gt = sns.pointplot(x=tmpz.index, y='NotRenwed%', data=tmpz, color='red', legend=True)
        gt.set_ylim(tmp['NotRenwed%'].min()-1,tmp['NotRenwed%'].max()+5)
        
        
    if thres is np.nan:
        gt.set_ylim(0,100)
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.set_ylim(0,100)
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
   


        
        
    #if thres is np.nan: 
    #    gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    #else:
    #    gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
    j=0
    for c in gt.collections:
        for i, of in zip(range(len(c.get_offsets())), c.get_offsets()):
            gt.annotate(values[j], of, color='brown', fontsize=12, rotation=45)
            j += 1
    plt.show()

    
    
def cat1_plot(data,col,p):    
#     plt.rcParams['figure.figsize'] = [15, 4]
#     plt.style.use("fivethirtyeight")
    if p == 1:
        print(data[col].value_counts(normalize=True,dropna=False))
    g = sns.countplot(data[col])
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,height + 3,'{}\n{:1.2f}%'.format(round(height),height/len(data)*100),ha="center", fontsize=10, color='blue')