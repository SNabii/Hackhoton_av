#!/usr/bin/env python
# coding: utf-8

# <p>Python Module <span style="color:blue;font-weight:bold">Importing Library</span> with the function written below <span style="color:darkolivegreen;font-weight:bold">dark green</span> eyes.</p>

# In[3]:

# basic import's
import os
import pandas as pd 
import numpy as np 
from tqdm import tqdm

# string 
import string
import re
import qgrid

pd.options.display.float_format = '{:.2f}'.format

# stats
import scipy.stats as stats
from scipy.stats import kurtosis, skew

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Nabi My-Functions ---------------------------------------------------------------------------------------

def getlist_nullcolumns(data):
    qq = data.isna().sum().reset_index()
    col_list = qq[qq[0] > 1]['index'].values
    col_list = col_list.tolist()
    return(col_list)

def get_partialcol_match(temp_df,col_text):
    date_colist = temp_df[temp_df.columns[temp_df.columns.to_series().str.contains(f'{col_text}')]].columns
    date_colist = date_colist.tolist()
    return(date_colist)

def do_flatlist(f0):
    x=[]
    for i in range(len(f0)):
        if 'list' in str(type(f0[i])):
            for j in range(len(f0[i])):
                x.append(f0[i][j])
        else :
            x.append(f0[i])
    #print(x)
    return(x)

def summarise_yourdf(df,leveli):
    print(f"Dataset Shape: {df.shape}")
    

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing_Count'] = df.isnull().sum().values
    summary['Missing_Perct'] = round(summary['Missing_Count']/df.shape[0]*100,2)
    summary['Uniques_Count'] = df.nunique().values
    summary['Uniques_Perct'] = round(summary['Uniques_Count']/df.shape[0]*100,2)
    
    #summary['First Value'] = df.loc[0].values
    #summary['Second Value'] = df.loc[1].values
    #summary['Third Value'] = df.loc[2].values
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    summary['Zeros_count'] = df[df == 0].count(axis=0).values
    summary['Zeros_Perct'] = round(summary['Zeros_count']/df.shape[0]*100,2)
    
    
    summary['Levels']= 'empty'
    for i, m in enumerate (summary['Name']):
            #print(i,m)
            if len(df[f'{m}'].value_counts()) <= leveli:
                #print(df[f'{m}'].value_counts())
                tab = df[f'{m}'].unique()
                summary.ix[i,'Levels']=f'{tab}'
    summary['N'] = df.shape[0]
    
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdf = df.select_dtypes(include=numerics)
    cols_nums =numdf.columns
    categoric = ['object']
    catdf = df.select_dtypes(include=categoric)
    cols_cat = catdf.columns#names of all the columns
    

    desc=df[cols_nums].describe().T
    desc = desc.reset_index()
    desc = pd.DataFrame(desc)
    desc.rename(columns={f'{desc.columns[0]}':'Name'}, inplace=True)
    desc.drop(['count'], axis=1,inplace=True)
    desc = round(desc,2)
   # desc

    merged_inner=pd.merge(summary, desc, on='Name', how='outer')
    merged_inner = merged_inner.replace(np.nan, '', regex=True)
    merged_inner = merged_inner.sort_values('Missing_Perct',ascending=False)
    #merged_inner.to_excel('4other_data/profiling.xlsx',index=False)
    #merged_inner.to_csv('4other_data/profiling.csv',index=False)
    return merged_inner

def grid_view(data):
    return(qgrid.show_grid(data,show_toolbar=True,grid_options={'forceFitColumns': False,'highlightSelectedCell': True,
        'highlightSelectedRow': True}))

def call_napercentage(data_train):
    op = pd.DataFrame(data_train.isnull().sum()/data_train.shape[0]*100)
    op = op.reset_index()
    op.rename(columns={'index':'variable_name'},inplace=True)
    op.rename(columns={0:'na%'},inplace=True)
    op=op.sort_values(by='na%',ascending=False)
    return(op)

def call_lfiles(path,ftype):
    import os
    import glob

    path = path#'D:/data/Analytics Track/Customer Retention Motor/Input_data2/csv/'
    extension = ftype#'csv'
    os.chdir(path)
    data_file = glob.glob('*.{}'.format(extension))
    return(data_file)

def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)




# Basic data cleaning
def data_basic_clean(fsh):
    fsh.columns = [c.strip() for c in fsh.columns]
    fsh.columns = [c.replace(' ', '_') for c in fsh.columns]
    fsh.columns = map(str.lower, fsh.columns)
    fsh.replace(['None','nan','Nan',' ','NaT','#REF!'],np.nan,inplace=True)
    fsh = trim_all_columns(fsh)
    fsh=fsh.drop_duplicates(keep='last')
    df = pd.DataFrame(fsh)
    return(df)


# Getting Single categorical distribution 

def get_univ_cat_distribution(data,target):
    f,ax=plt.subplots(1,2,figsize=(15,6))
    data[f'{target}'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title(f'{target}')
    ax[0].set_ylabel('')
    sns.countplot(f'{target}',data=data,ax=ax[1])
    ax[1].set_title(f'{target}')
    plt.show()
    
def plot_count(feature, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show() 
    
def get_numcolnames(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdf = df.select_dtypes(include=numerics)
    cols_nums =numdf.columns
    cols_nums = cols_nums.tolist()
    return(cols_nums)

def get_catcolnames(df):
    categoric = ['object']
    catdf = df.select_dtypes(include=categoric)
    cols_cat = catdf.columns
    cols_cat = cols_cat.tolist()
    return(cols_cat)


def call_catscore(df,cat_col,y_lab,Top_n,thres,ytag,prt):
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

    tmpz=pd.merge(tmp,tmp1,left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = tmpz['Renewed%'].mean()
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    tmpz['score'] = round(tmpz['R_count']/tmpz['Tot'],2)
    #tmpz.sort_index(inplace=True)
    
    # Statistic calculation ------
    score_mean = tmpz['score'].mean()
    score_std = tmpz['score'].std()
    th_min = tmpz['score'].min()
    th_nsd = round(score_mean - score_std,2)
    th_mean = round(score_mean,2)
    th_psd = round(score_mean + score_std,2)
    th_max = tmpz['score'].max()
    
    def producer_clfy(tmpz):
        if (tmpz['score'] >= th_min and tmpz['score'] < th_nsd):
            return 'c4'
        if (tmpz['score'] >= th_nsd and tmpz['score'] < th_mean):
            return 'c3'
        elif (tmpz['score'] >= th_mean and tmpz['score'] < th_psd):
            return 'c2'
        elif (tmpz['score'] >= th_psd and tmpz['score'] <= 1):
            return 'c1'
        elif (tmpz['score'] > 1):
            return np.nan
    tmpz[f'{i}_class'] = tmpz.apply(producer_clfy, axis = 1)
    tmpz.reset_index(inplace = True)
    tmpz.rename(columns={f'{i}':'count'}, inplace=True)
    tmpz.rename(columns={'index':f'{i}'}, inplace=True)
    
    #tmpz = tmpz[[f'{i}',f'{i}_class','score']]
    
    #tmpz.drop('index',axis=1,inplace=True)
    
    #-----------------------------
    if prt == 'Y':
        print(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        tmpzi.to_excel("tmpz.xlsx")
    return(tmpz)

def do_response_scoring(df,xlab,ylab,Top_n,prt):    
    df_ins = df # data frame 
    #columns = ['producer_cd','veh_make_name','veh_mdl_name'] # grouping of 
    columns = xlab #=['veh_make_name'] # grouping of 
    columns_conc = ("|".join(columns))
    #columns_conc

    df_ins[f'{columns_conc}'] = df_ins[columns].astype(str).astype(str).apply('|'.join, axis=1)
    #grid_view(df_ins.head())

    i = cat_col = columns_conc
    y2 = ylab #="target"
    Top_n = Top_n #=1500000000000000000000000000000000000000000000000
    ytag = ytag = 1
    col_count  = df_ins[f'{i}'].value_counts()
    #print(col_count)
    col_count = col_count[:Top_n,]


    col_count1 = df_ins[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp =pd.crosstab(df_ins[f'{i}'], df_ins[f'{y2}'], normalize='index') * 100
    #tmp.head(5)

    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    #tmp.head(5)

    tmp = pd.DataFrame(tmp)
    #tmp.columns

    tmp.rename(columns={'0':'NotRenwed%', '1':'Renewed%'}, inplace=True)

    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0

    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={'0':'NR_count', '1':'R_count'}, inplace=True)
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
    tmpz['score'] = round(tmpz['R_count']/tmpz['Tot'],2)
    #tmpz.sort_index(inplace=True)

    # Statistic calculation ------
    score_mean = tmpz['score'].mean()
    score_std = tmpz['score'].std()
    th_min = tmpz['score'].min()
    th_nsd = round(score_mean - score_std,2)
    th_mean = round(score_mean,2)
    th_psd = round(score_mean + score_std,2)
    th_max = tmpz['score'].max()

    tmpz=pd.merge(tmp,tmp1,left_index=True,right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = tmpz['Renewed%'].mean()
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    tmpz['score'] = round(tmpz['R_count']/tmpz['Tot'],2)
    tmpz = tmpz.reset_index()
    tmpz.columns.values[tmpz.columns.get_loc(f'{columns_conc}')] = 'count'
    tmpz.rename(columns={'index':f'{columns_conc}','score':f'{columns_conc}_score'},inplace=True)
    
    if len(xlab) > 1:
        tmpz = tmpz.join(pd.DataFrame(tmpz[f'{tmpz.columns.values[0]}'].str.split('|').tolist()))
        namu = tmpz.columns.values[0]
        namu = namu.split('|')
        cl = dict(enumerate(namu))
        tmpz.rename(columns=cl, inplace=True)
        
        
        
    select_reqcolumn = []
    select_reqcolumn.append(xlab)
    columns = xlab #=['veh_make_name'] # grouping of 
    columns_conc = ("|".join(columns))
    score_col = [f'{columns_conc}_score']
    select_reqcolumn.append(score_col)
    select_reqcolumn = do_flatlist(select_reqcolumn)
    get_scoredf = tmpz[select_reqcolumn]
    tmpz = get_scoredf
    
    
    
    if prt == 'Y':
        tmpz.to_csv(f'{columns_conc}.csv',index=False)
    return tmpz

def detect_outliers(variable):
    global filtered
    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate lower fence and upper fence for outliers
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.
    
    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    print('Total Outliers of', variable.name,':', outliers.count())
    
    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis = 0)

    # Create subplots
    out_variables = [variable, filtered]
    out_titles = [' Distribution with Outliers', ' Distribution Without Outliers']
    title_size = 25
    font_size = 18
    plt.figure(figsize = (25, 15))
    for ax, outlier, title in zip(range(1,3), out_variables, out_titles):
        plt.subplot(2, 1, ax)
        sns.boxplot(outlier).set_title('%s' %outlier.name + title, fontsize = title_size)
        plt.xticks(fontsize = font_size)
        plt.xlabel('%s' %outlier.name, fontsize = font_size)

def get_outlier_analysis(df,span):

    df1 = df
    span = span #5 
    index_col = get_numcolnames(df1)
    def gen_list(start, stop,step):
        return ['%sptile' % i for i in range(start, stop,step)]
    colnums = gen_list(0,102,span)
    temp_df = pd.DataFrame(columns=colnums,index=index_col)
    print(temp_df.shape)

    for irow in range(0,temp_df.shape[0]):
        print(temp_df.index[irow],":",irow,"/",temp_df.shape[0])

        temp_list = []
        for i in range(0,100,span):
            var =df1[f"{temp_df.index[irow]}"].values
            var = np.sort(var,axis = None)
            temp_list.append(var[int(len(var)*(float(i)/100))])
        temp_list.append(var[-1])
        #print(len(temp_list))

        for ifeed in range(0,len(temp_list)):
            #print(icol,"with content : ",temp_list[icol])
            temp_df.iloc[irow,ifeed] = temp_list[ifeed]
    temp_df.to_excel('outlier_report.xlsx',index=True)
    return temp_df
    
def concat_columns(xlab):
    columns = xlab #=['ncb','age_of_vehicle'] # grouping of 
    columns_conc = ("|".join(columns))
    temp_df[f'{columns_conc}_combi'] = temp_df[columns].astype(str).astype(str).apply('|'.join, axis=1)
    return(temp_df)

def reduce_catgory(data,coli,breaki,roundi):
    # coli =  'retail_city'
    # breaki = 5
    # roundi = 1
    df2 = data 

    try:
        vi = df2.groupby([f'{coli}','target']).size().reset_index().pivot(index=f'{coli}', columns='target', values=0)
    except:
        vi = pd.crosstab(temp_df[f'{coli}'], temp_df['target'])
    #vi.rename(columns={0:'no',1:'yes'},inplace=True)
    vi = vi.reset_index()
    vi.columns=[f'{coli}','no','yes']
    vi.fillna(0,inplace=True) 
    vi['total'] =  vi['no']+vi['yes']

    vi['no_score'] =  round((vi['no']/vi['total']),4)
    vi['yes_score'] =  round((vi['yes']/vi['total']),4)

    #vi['total_qcut'] = pd.qcut(vi['yes_score'], breaki)
    vi['yes_score']  = round(vi['yes_score'],roundi)
    #vi.sort_values(by='total_qcut')

    from itertools import groupby
    N = vi['yes_score'].tolist()
    slist = [list(j) for i, j in groupby(N)]
    #print(slist)

    ti=len(slist)

    temp_list = []
    for i in slist:
        #print(i[0])
        temp_list.append(i[0])

    temp_list=set(temp_list)
    temp_list =list(temp_list)

    temp_list = temp_list

    #len(set(temp_list))

    tr = pd.DataFrame()
    tr['cut_val'] = temp_list
    tr['cut_val']=tr['cut_val'].astype('float')
    tr = tr.sort_values(by='cut_val',ascending=False)



    collecti = []
    for i in range(1,(tr.shape[0]+1)):
        collecti.append(f'w{i}')


    tr['level'] = collecti
    #print(tr)

    area_dict = dict(zip(tr['cut_val'], tr['level']))
    #print(area_dict)

    vi[f'{coli}_level'] = vi['yes_score'].map(area_dict)
    vi.sort_values(by='yes_score',ascending=False,inplace=True)
#     vt = temp_riz.groupby(['retail_city_level']).agg({'yes_score':['count','min','max']})
#     vt = vt.reset_index()
#     vt.columns =  ['yes_score','count','min','max']
#     vt.sort_values(by='max',ascending=False)

    return(vi)


def call_cattarget(temp_riz,columns_conc):
    vt = temp_riz.groupby([f'{columns_conc}']).agg({'target':['count','min','max']})
    vt = vt.reset_index()
    vt.columns =  ['yes_score','count','min','max']
    vt = vt.sort_values(by='count',ascending=False)
    return(vt)

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    elif n == 1:
        return([start])
    else:
        return([])
    
    
def call_outanalysis_wo_act(id_col,y_pred,y_prob):
    
    out_nov = pd.DataFrame(id_col)
    out_nov['y_pred'] =  pd.Series(y_pred)
    out_nov['y_prob'] =  pd.Series(y_prob)
    #out_nov['y_act'] =  pd.Series(y_test_nov)
    bins = seq(0,1,0.05)
    out_nov['prob_bin'] = pd.cut(out_nov['y_prob'], bins)
    ti = out_nov.groupby(['prob_bin']).size()
    ti = pd.DataFrame(ti)
    ti.reset_index(inplace=True)
    ti=ti.sort_values(by='prob_bin',ascending=False)
    ti.columns=['Group_score','policy_count']
    ti['cum%'] = (ti['policy_count'].cumsum()/(ti['policy_count'].sum()))*100
    ti['%'] = (ti['policy_count']/ti['policy_count'].sum())*100
    return(ti)

def call_outanalysis_wi_act(id_col,y_pred,y_prob,y_act):
    
    temp_df = pd.DataFrame(id_col)
    temp_df['y_pred'] =  pd.Series(y_pred)
    temp_df['y_prob'] =  pd.Series(y_prob)
    temp_df['y_act'] =  pd.Series(y_act)
    bins = seq(0,1,0.05)
    temp_df['prob_bin'] = pd.cut(temp_df['y_prob'], bins)
    
    act =temp_df.groupby(['prob_bin','y_act']).size().reset_index().pivot(index='prob_bin', columns='y_act', values=0)
    act = act.reset_index()
    act.columns=['prob_bin','no','yes']
    #act.fillna('0',inplace=True) 
    act['total'] =  act['no']+act['yes']
    act=act.sort_values(by='prob_bin',ascending=False)
    act['total%'] = (act['total']/act['total'].sum())*100
    act['cum%'] = (act['total'].cumsum()/act['total'].sum())*100
    act['no%'] = (act['no']/act['total'])*100
    act['yes%'] = (act['yes']/act['total'])*100
    return(act)

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from itertools import product, chain
from tqdm import tqdm

RANDOM_STATE = 0
def catboost_GridSearchCV(X, y, X_test, params, cat_features, n_splits=5):
    ps = {'acc':0,
          'param': []
    }
    
    predict=None
    
    for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):
                          
        acc = cross_val(X, y, X_test, prms, cat_features, n_splits=5)

        if acc>ps['acc']:
            ps['acc'] = acc
            ps['param'] = prms
    print('Acc: '+str(ps['acc']))
    print('Params: '+str(ps['param']))
    
    return ps['param']

def cross_val(X, y, X_test, param, cat_features, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    acc = []
    predict = None
    
    for tr_ind, val_ind in skf.split(X, y):
        X_train = X.iloc[tr_ind]
        y_train = y.iloc[tr_ind]
        
        X_valid = X.iloc[val_ind]
        y_valid = y.iloc[val_ind]
        
        clf = CatBoostClassifier(iterations=500,
                                loss_function = param['loss_function'],
                                depth=param['depth'],
                                l2_leaf_reg = param['l2_leaf_reg'],
                                eval_metric = 'Accuracy',
                                leaf_estimation_iterations = 10,
                                use_best_model=True,
                                logging_level='Silent'
        )
        
        clf.fit(X_train, 
                y_train,
                cat_features=cat_features,
                eval_set=(X_valid, y_valid)
        )
        
        y_pred = clf.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        acc.append(accuracy)
    return sum(acc)/n_splits