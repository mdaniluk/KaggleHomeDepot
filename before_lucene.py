# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:32:27 2016

@author: dudu
"""

import pandas as pd
import os
from preprocess_forum_stem import str_stem

def preprocess_queries():
    df_train = pd.read_csv('data/train.csv')
    df_train['search_term'] = df_train['search_term'].map(lambda x:str_stem(x))
    df_train.to_csv('AllenLucene/data/train.csv', index=False)
    df_test = pd.read_csv('data/test.csv')
    df_test['search_term'] = df_test['search_term'].map(lambda x:str_stem(x))
    df_test.to_csv('AllenLucene/data/test.csv', index=False)
    

def load_products():
    df_pro_desc = pd.read_csv('data/product_descriptions.csv')
    df_attr = pd.read_csv('data/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"]\
    [["product_uid", "value"]].rename(columns={"value": "brand"})
    
    df_db = pd.merge(df_pro_desc, df_brand, how='left', on='product_uid')
    print 'data loaded'
    return (df_db, df_attr)
 
   
def create_files_for_lucene(df_db, df_attr, dist_dir='AllenLucene/data/files'):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
#    count = 0
    for _, row in df_db.iterrows():
        uid = row['product_uid']
        attr_list = [str(row['brand']), str(row['product_description'])]
        for _, a in df_attr[df_attr.product_uid == uid].iterrows():
            attr_list.append(' : '.join([str(a['name']), str(a['value'])]))
        content = ('\n'.join(attr_list))
        
        out = open(os.path.join(dist_dir, str(uid)), 'w')
        out.write(str_stem(content))
        out.close
#        count += 1
#        if count == 10:
#            break
        
if __name__ == '__main__':
    preprocess_queries()
    df_db, df_attr = load_products()
    create_files_for_lucene(df_db, df_attr)
    


