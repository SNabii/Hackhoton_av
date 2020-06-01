#!/usr/bin/env python
# coding: utf-8

# <p>Python Module <span style="color:blue;font-weight:bold">Importing Library</span> with the function written below <span style="color:darkolivegreen;font-weight:bold">dark green</span> eyes.</p>

# In[3]:

# basic import's
import os
import pandas as pd 
import numpy as np 
import re
import string
import qgrid
# stats
import scipy.stats as stats
from scipy.stats import kurtosis, skew

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Nabi My-Functions ---------------------------------------------------------------------------------------

def initial_filters(temp_df):
    
    try:
        temp_df['discount'] = np.where(temp_df['discount'].isna(),0,temp_df['discount'])
        temp_df['discount']=temp_df['discount'].astype('int')

    except:

        try:
            temp_df['discount'] =temp_df['discount'].str.replace('%','')
        except:
            temp_df['discount'] =temp_df['discount']*100
            temp_df['discount'] = np.where(temp_df['discount'].isna(),0,temp_df['discount'])
    
    
    temp_df['discount'] = np.where(temp_df['discount'].isna(),0,temp_df['discount'])
    temp_df['inception_month'] = pd.to_datetime(temp_df['inception_month'], format='%b-%y').map(lambda x: x.strftime('%Y-%m'))
    temp_df['pctchange']  = pd.to_numeric(temp_df['pctchange'], errors='coerce')
    temp_df=temp_df[temp_df['package_name'] != 'LiabilityOnly']
    temp_df=temp_df[temp_df['product_name'] == 'Private Car Insurance']
    temp_df = temp_df[temp_df['producer_name'] != 'TATA MOTORS INSURANCE BROKING AND ADVISORY SERVICES LTD'] # Excluding TMI
    temp_df= temp_df[(temp_df['certificate_no'] >= 1.0) & (temp_df['certificate_no'] <= 10.0)] # consider certificate 
    temp_df['ncb']  = np.where((temp_df['ncb'].isnull()) & (temp_df['package_name'] == 'StandAloneOD'), temp_df['expiring_ncb'], temp_df['ncb'])
    temp_df['veh_make_name'].replace(to_replace = ['TOY', 'TOYOT', 'TOYOTA', 'TOYOTO'], value = 'TOYOTA', inplace = True)  
    
    
    temp_df['discount'] =temp_df['discount'].astype('int64')
    return(temp_df)


def call_phone_email(tempi):
    tempi = tempi[['email_id','lead_phone1_flag']]
    tempi['eml_status'] = np.where(tempi['email_id'].isna(),1,0)
    tempi[~tempi['email_id'].isna()].head()

    tempi['email_id'] = np.where(tempi['email_id'] == 0.0 , np.nan ,tempi['email_id'] )

    def domainsplit(x):
        try:
            return (x.split('@')[1])
        except:
            return ('nodomain')

    tempi['domain'] = tempi['email_id'].apply(lambda x: domainsplit(x))

    tempi['domain'].value_counts(sort=True)[0:10]

    tempi['both_flag'] = np.where((tempi['lead_phone1_flag'] == 1) & (tempi['domain'] != 'nodomain'), 1, 0)

    tempi['all_flag'] = np.where((tempi['lead_phone1_flag'] == 1) & (tempi['eml_status'] == 1), 'phone_email', 
                            np.where((tempi['lead_phone1_flag'] == 1) & (tempi['eml_status'] == 0), 'onlyphone', 
                                    np.where((tempi['lead_phone1_flag'] == 0) & (tempi['eml_status'] == 1), 'onlyemail', 'others' )))
    return(tempi['all_flag'])

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


def create_binary_variables(data):
    data['od_ind'] = np.where(data['od_premium'] > 0,1,0)
    data['tppd_ind'] = np.where(data['premium_tppd'] > 0,1,0)
    data['driver_ind'] = np.where(data['premium_driver'] > 0,1,0)
    data['personal_accident_ind'] = np.where(data['premium_personal_accident'] > 0,1,0)
    return data

def get_partialcol_match(final_df,txt):
    date_colist = final_df[final_df.columns[final_df.columns.to_series().str.contains(f'{txt}')]].columns
    date_colist = date_colist.tolist()
    return(date_colist)
def get_contactflag(temp_df):
    lp_cols = get_partialcol_match(temp_df,'lead_phone')
    
    def rm_specialchar(stri):
        stri  = str(stri)
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        out = regex.sub('', stri)    
        return(out)

    for coli in lp_cols:
        temp_df[f'{coli}'] = np.where(temp_df[f'{coli}'].isna(),'0',temp_df[f'{coli}'])
        temp_df[f'{coli}'] = temp_df[f'{coli}'].apply(rm_specialchar)
        temp_df[f'{coli}'] = np.where(((temp_df[f'{coli}'] == '0') | (temp_df[f'{coli}'] == '00')),np.nan , temp_df[f'{coli}'])


    print(temp_df[~temp_df['lead_phone1'].isna()].head())

    temp_df['lead_phone1_flag'] = np.where(temp_df['lead_phone1'].isna() , 0 , 1 )
    temp_df  = temp_df[temp_df.columns.difference(lp_cols)]
    print('****Data Dimension****',temp_df.shape)

    temp_df['contact_flag']=call_phone_email(temp_df)

    #email_id lead_phone1_flag
    temp_df.drop(columns=['email_id','lead_phone1_flag'],axis=1,inplace=True)
    return(temp_df)


def last_nafill(dataset):
    
        for coli in dataset.columns:

            if(dataset[f'{coli}'].dtypes == 'object'):
                #print(coli)
                #print(dataset[f'{coli}'].value_counts(dropna=False))
                dataset[f'{coli}'].fillna('others', inplace=True)
                #print(coli,"is object")
                #print('-'*70)

            if(dataset[f'{coli}'].dtypes != 'object'):
                #print(coli)
                #print(dataset[f'{coli}'].value_counts(dropna=False))
                dataset[f'{coli}'].fillna(0, inplace=True)
                #print(coli,"is Not object")
                #print('-'*70)
        return(dataset)
def check_samecolnames(pand_df):
    er = pd.DataFrame(columns = ['Col_nam'])
    er['Col_nam'] = pand_df.columns
    er=er.groupby('Col_nam')['Col_nam'].agg(['count']).sort_values(['count'],ascending=False)
    er.reset_index()
    #er.head(10) #.apply(pd.DataFrame.sort, 'count')
    return(er)
def rename_duplicate_col(duplicate_col):
    s = duplicate_col.columns.to_series()
    new = s.groupby(s).cumcount().astype(str).radd('_').replace('_0','')
    duplicate_col.columns += new
    return(duplicate_col)
    