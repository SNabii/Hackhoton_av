#!/usr/bin/env python
# coding: utf-8

# <p>Python Module <span style="color:blue;font-weight:bold">Importing Library</span> with the function written below <span style="color:darkolivegreen;font-weight:bold">dark green</span> eyes.</p>

# In[3]:

import os
import pandas as pd 
import numpy as np 
import scipy.stats as stats
# import qgrid
from scipy.stats import kurtosis, skew
import seaborn as sns
import matplotlib.pyplot as plt

# import pyarrow.parquet as pq
# import pyarrow as pa
import pandas as pd
import numpy as np
from scipy import interp

# from dplython import select, DplyFrame, X, arrange, count, sift, head, summarize, group_by, tail, mutate

from sklearn.model_selection import validation_curve
import sklearn.metrics as metrics

from sklearn.metrics import auc, accuracy_score
from sklearn.metrics  import plot_roc_curve
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from itertools import product, chain
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

def initial_filters(raw_df):    
    
    raw_df['lead_phone1'] = raw_df['lead_phone1'].astype('str')
    raw_df['lead_phone1'] = pd.to_numeric(raw_df['lead_phone1'], errors='coerce')
    raw_df['lead_phone1_len'] = np.where(raw_df['lead_phone1']==0,np.nan,raw_df['lead_phone1'])
    raw_df['lead_phone1_flag'] = np.where(raw_df['lead_phone1_len'].isna(),0,1)
    raw_df = raw_df.drop(['lead_phone1'], axis=1)
    #raw_df['email_id'] = raw_df['email_id'].astype('str')

    raw_df = raw_df.drop(['lead_phone1_len'], axis=1)
    raw_df['veh_make_name'].replace(to_replace = ['TOY', 'TOYOT', 'TOYOTA', 'TOYOTO'], value = 'TOYOTA', inplace = True)
    
    raw_df = raw_df[~raw_df['veh_mdl_name'].isna()]
    #raw_df = raw_df[~raw_df['premium_named_personal_acc'].isna() & ~raw_df['od_premium'].isna() & ~raw_df['discount'].isna()& ~raw_df['premium_tppd'].isna()  & ~raw_df['premium_driver'].isna()& ~raw_df['premium_new_owndamage'].isna()& ~raw_df['premium_tppd'].isna()]
    
    raw_df = raw_df[raw_df['producer_name'] != 'TATA MOTORS INSURANCE BROKING AND ADVISORY SERVICES LTD'] # Excluding TMI
    raw_df= raw_df[(raw_df['certificate_no'] >= 1.0) & (raw_df['certificate_no'] <= 10.0)] # consider certificate 
    raw_df['ncb']  = np.where((raw_df['ncb'].isnull()) & (raw_df['package_name'] == 'StandAloneOD'), raw_df['expiring_ncb'] , raw_df['ncb'])
    raw_df['ncb']  = np.where(raw_df['ncb'].isnull(), 0.0 , raw_df['ncb'])
    raw_df['curr_plan_type'] = np.where(raw_df['curr_plan_type'].isnull(), 'others' , raw_df['curr_plan_type'])
    raw_df['rta_city'] = np.where(raw_df['rta_city'].isnull(), 'others' , raw_df['rta_city'])
    raw_df['fuel_type'] = np.where(raw_df['fuel_type'].isnull(), 'others' , raw_df['fuel_type'])
    raw_df['break'] = np.where(raw_df['break'].isnull(), 'others' , raw_df['break'])

    #raw_df['package_name'] = raw_df[raw_df['package_name'] != 'LiabilityOnly']
    #raw_df['product_name'] = raw_df[raw_df['product_name'] == 'Private Car Insurance']
    # Getting date columns --------------
    #raw_df['inception_month'] = pd.to_datetime(raw_df['inception_month'], format='%b-%y').map(lambda x: x.strftime('%Y-%m'))
    # date_colist = raw_df[raw_df.columns[raw_df.columns.to_series().str.contains('date')]].columns
    # date_colist = date_colist.tolist()
    # print(date_colist)
    # grid_view(raw_df[date_colist].head())

    #raw_df['rating_zone'] = np.where(raw_df['rating_zone'].isna()  ,'others' ,raw_df['rating_zone'])
    raw_df['veh_mdl_name'] = np.where(raw_df['veh_mdl_name'].isna()  ,'others' ,raw_df['veh_mdl_name'])
   
    
    return(raw_df)




# Iq ----------------------------------------------------------------------------------------------------------------------

def target_encoder(df, column, target, index=None, method='mean'):
   
    index = df.index if index is None else index # Encode the entire input df if no specific indices is supplied

    if method == 'mean':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].mean())
    elif method == 'median':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].median())
    elif method == 'std':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].std())
    else:
        raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median', 'std'".format(method))

    return encoded_column

def smoothing_target_encoder(df, column, target, weight=100):
 
    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(column)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the 'smoothed' means
    smooth = (counts * means + weight * mean) / (counts + weight)
    smooth = smooth.to_dict()
#     print('*** Smoothed Mean',smooth)
    # Replace each value by the according smoothed mean
    return df[column].map(smooth)

def read_data(filename):
    path='/home/jovyan/jupyter/notebooks/AIG_Retention/'
#     data = pd.read_csv(path+'Retention_data.csv')
    data = pd.read_csv(path+filename)
    
    ## Filter by certificate numbers - Incorrect data
    data = data[(data['certificate_no'] >= 1) & (data['certificate_no'] <= 10)]
    
    # Filter on veh_mdl_name to remove NaN's from the column
    data = data[~data['veh_mdl_name'].isna()]
    #data = data[~data['premium_named_personal_acc'].isna() & ~data['od_premium'].isna() & ~data['discount'].isna()& ~data['premium_tppd'].isna() & ~data['premium_driver'].isna()& ~data['premium_new_owndamage'].isna()& ~data['premium_tppd'].isna()]
    
    
    return data


def imputation_strategy1(df,within_group,columns):
    print('**** Deriving Grouped Averages  ****')
    imputed_df =  df[columns+within_group].groupby(within_group).median().reset_index()
    
    print('**** Merging the Data & Derived Averages  ****')
    df = df.merge(imputed_df,left_on=within_group, right_on=within_group,suffixes=('', '_enc'))
    
    df['od_premium'] = np.where(df['od_premium']==0,df['od_premium_enc'],df['od_premium'])
    df['premium_named_personal_acc'] = np.where(df['premium_named_personal_acc']==0,
                                                        df['premium_named_personal_acc_enc'],df['premium_named_personal_acc'])
    df['premium_tppd'] = np.where(df['premium_tppd']==0,df['premium_tppd_enc'],df['premium_tppd'])
    df['premium_driver'] = np.where(df['premium_driver']==0,df['premium_driver_enc'],df['premium_driver'])
    df['premium_new_owndamage'] = np.where(df['premium_new_owndamage']==0,df['premium_new_owndamage_enc']
                                           ,df['premium_new_owndamage'])    
    #df = df.drop(['od_premium','premium_named_personal_acc','premium_tppd','premium_driver','premium_new_owndamage'])
    
    return df

def imputation_strategy2(df):
    categorical_var = df.select_dtypes(include ='object').columns
    df[categorical_var] = df[categorical_var].fillna('others')
    df['pctchange'] = df['pctchange'].fillna(0.0)
    #df['ncb']  = np.where((df['ncb'].isnull()) & (df['package_name'] == 'StandAloneOD'), df['expiring_ncb'] , df['ncb'])
    df['ncb']=df['ncb'].fillna(0)
    #df['ncb']  = np.where(df['ncb'].isnull(), 0.0 , df['ncb'])
    
    return df

def create_binary_variables(data):
    data['od_ind'] = np.where(data['od_premium'] > 0,1,0)
    data['tppd_ind'] = np.where(data['premium_tppd'] > 0,1,0)
    data['driver_ind'] = np.where(data['premium_driver'] > 0,1,0)
    data['personal_accident_ind'] = np.where(data['premium_personal_accident'] > 0,1,0)
    return data

def encode_categorical_var(df):
    categorical_var = list(df.select_dtypes(exclude ='float').columns)
    categorical_var.remove('target')
    model_df = new_df.copy()
    for col in categorical_var:
        model_df[col] =  smoothing_target_encoder(new_df, col, 'target')
         
    return model_df

def read_data_new(folder, file_name):
    df = pd.read_csv(os.path.join(folder, file_name))
    return df 





# Iq ------------------------------------------------------------------------------------------------------------------Ends



































# ----$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$




# In[ ]:

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,5)):
    
    from sklearn.metrics import confusion_matrix
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=None)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    
    
#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm[i, j]
#             p = cm_perc[i, j]
#             if i == j:
#                 s = cm_sum[i]
#                 annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
#             elif c == 0:
#                 annot[i, j] = ''
#             else:
#                 annot[i, j] = '%.1f%%\n%d' % (p, c)
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
#                 annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                #annot[i, j] = '%.1f%%\n%d' % (p, c)
                #annot[i, j] = '%d\n%d' % (c,T)
                annot[i, j] = '%d' % (c)
            elif c == 0:
                annot[i, j] = ''
            else:
                #annot[i, j] = '%.1f%%\n%d' % (p, c)
                #annot[i, j] = '%d\n%d' % (c,T)
                annot[i, j] = '%d' % (c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

def getlist_nullcolumns(data):
    qq = data.isna().sum().reset_index()
    col_list = qq[qq[0] > 1]['index'].values
    col_list = col_list.tolist()
    return(col_list)

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
    merged_inner.to_excel('profiling.xlsx',index=False)
    return merged_inner

def grid_view(data):
    return(qgrid.show_grid(data,show_toolbar=True,grid_options={'forceFitColumns': False,'highlightSelectedCell': True,
        'highlightSelectedRow': True
}))

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



# Word Cloud Plotting  ---------

# from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# stopwords = set(STOPWORDS)

# def show_wordcloud(data, title = None):
#     wordcloud = WordCloud(
#         background_color='white',
#         stopwords=stopwords,
#         max_words=200,
#         max_font_size=40, 
#         scale=3,
#         random_state=1 # chosen at random by flipping a coin; it was heads
#     ).generate(str(data))

#     fig = plt.figure(1, figsize=(12, 12))
#     plt.axis('off')
#     if title: 
#         fig.suptitle(title, fontsize=20)
#         fig.subplots_adjust(top=2.3)

#     plt.imshow(wordcloud)
#     plt.show()
    
# Basic Data Cleaninf Function 

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
    
    