# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:16:03 2016

@author: dudu
"""

import numpy as np
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer

def load_train_and_desc():
    df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
#    df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('data/product_descriptions.csv')
    df_attr = pd.read_csv('data/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"]\
    [["product_uid", "value"]].rename(columns={"value": "brand"})
#    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = df_train
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    
    df_db = pd.merge(df_pro_desc, df_brand, how='left', on='product_uid')
    print 'data loaded'
    return (df_all, df_db, df_attr)
    
def create_files_for_lucene(df_db, df_attr, dist_dir='data/files'):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    for _, row in df_db.iterrows():
        uid = row['product_uid']
        attr_list = [str(row['brand']), str(row['product_description'])]
        for _, a in df_attr[df_attr.product_uid == uid].iterrows():
            attr_list.append(' : '.join([str(a['name']), str(a['value'])]))
        content = ('\n'.join(attr_list))
        
        out = open(os.path.join(dist_dir, str(uid)), 'w')
        out.write(content)
        out.close
        
if __name__ == '__main__':
    df_all, df_db, df_attr = load_train_and_desc()
    create_files_for_lucene(df_db, df_attr)
        

        
        
        
            
        
    
        
        
    
    
    
