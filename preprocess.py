# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:16:03 2016

@author: dudu
"""

import numpy as np
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

stop = stopwords.words('english')

def load_train_and_desc():
    df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
#    df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
    df_lucene_features = pd.read_csv('AllenLucene/data/lucene_features_stem.csv', \
        names=['id', 'in_top_lucene', 'lucene_ranking_place', 'lucene_score'])
    df_pro_desc = pd.read_csv('data/product_descriptions.csv')
    df_attr = pd.read_csv('data/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"]\
    [["product_uid", "value"]].rename(columns={"value": "brand"})
#    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = df_train
    df_all = pd.merge(df_all, df_lucene_features, how='left', on='id')
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    
    df_db = pd.merge(df_pro_desc, df_brand, how='left', on='product_uid')
    print 'data loaded'
    return (df_all, df_db, df_attr)
    
def create_files_for_lucene(df_db, df_attr, dist_dir='AllenLucene/data/files'):
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
        
def get_attributes_as_text(uid, df_attr):
    attr_list = []
    for _, a in df_attr[df_attr.product_uid == uid].iterrows():
        attr_list.append(' '.join([str(a['name']), str(a['value'])]))
    content = (', '.join(attr_list))
    return preprocess_text(content.decode('utf8'))
        
def create_ngrams_bag(text, n):
    t = text.split()
    bag = []
    for i in range(len(t)+1-n):
        bag.append(' '.join(t[i:i+n]))
    return bag
        
def count_common_ngrams(query, text, n):
    q_bag = set(create_ngrams_bag(query, n))
    t_bag = set(create_ngrams_bag(text, n))
    common = q_bag.intersection(t_bag)
    if len(q_bag) == 0:
        return 0.0
    else :
        return float(len(common)) / float(len(q_bag))
        
def calculate_avg_tf(query, text, ngram):
    q_bag = set(create_ngrams_bag(query, ngram))
    t_list = create_ngrams_bag(text, ngram)
    if (len(t_list) == 0) or (len(q_bag) == 0):
        return 0.0
    s = 0.0
    for q in q_bag:
        s += t_list.count(q)
    s /= float(len(q_bag))
    return float(s) / len(t_list)
    
    
def preprocess_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    processed_text = []
    for w in words:
        if w not in stop:
            processed_text.append(stemmer.stem(w.lower()))
    return ' '.join(processed_text)
    
        
if __name__ == '__main__':
    df_all, df_db, df_attr = load_train_and_desc()
#    create_files_for_lucene(df_db, df_attr)
        
#    df_all = df_all[:10]
    df_all['lucene_ranking_place'] = \
        (201 - df_all['lucene_ranking_place']) / 200.0
    df_all['prepr_attrs'] = \
        df_all['product_uid'].map(lambda x: get_attributes_as_text(x, df_attr))
    df_all['prepr_descr'] = \
        df_all['product_description'].map(lambda x: preprocess_text(x.decode('utf8')))
    df_all['prepr_query'] = \
        df_all['search_term'].map(lambda x: preprocess_text(x.decode('utf8')))
    df_all['prepr_title'] = \
        df_all['product_title'].map(lambda x: preprocess_text(x.decode('utf8')))
    df_all['prepr_brand'] = \
        df_all['brand'].map(lambda x: \
        preprocess_text(x.decode('utf8')) if x != None else '')
    df_all['query_title_common_1_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_title'],1), \
        axis=1)
    df_all['query_title_common_2_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_title'],2), \
        axis=1)
    df_all['query_descr_common_1_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_descr'],1), \
        axis=1)
    df_all['query_descr_common_2_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_descr'],2), \
        axis=1)
    df_all['query_attrs_common_1_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_attrs'],1), \
        axis=1)
    df_all['query_attrs_common_2_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_attrs'],2), \
        axis=1)
    df_all['query_brand_common_1_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_brand'],1), \
        axis=1)
    df_all['query_brand_common_2_gram'] = \
        df_all.apply(lambda x: \
        count_common_ngrams(x['prepr_query'], x['prepr_brand'],2), \
        axis=1)
    df_all['query_title_avg_1_gram_tf'] = \
        df_all.apply(lambda x: \
        calculate_avg_tf(x['prepr_query'], x['prepr_title'],1), \
        axis=1)
    df_all['query_title_avg_2_gram_tf'] = \
        df_all.apply(lambda x: \
        calculate_avg_tf(x['prepr_query'], x['prepr_title'],2), \
        axis=1)
    df_all['query_descr_avg_1_gram_tf'] = \
        df_all.apply(lambda x: \
        calculate_avg_tf(x['prepr_query'], x['prepr_descr'],1), \
        axis=1)
    df_all['query_descr_avg_2_gram_tf'] = \
        df_all.apply(lambda x: \
        calculate_avg_tf(x['prepr_query'], x['prepr_descr'],2), \
        axis=1)
        
    df_features = df_all[['id', 'lucene_ranking_place', 'lucene_score',\
    'query_title_common_1_gram', 'query_title_common_2_gram', \
    'query_descr_common_1_gram', 'query_descr_common_2_gram', \
    'query_attrs_common_1_gram', 'query_attrs_common_2_gram', \
    'query_brand_common_1_gram', 'query_brand_common_2_gram', \
    'query_title_avg_1_gram_tf', 'query_title_avg_2_gram_tf', \
    'query_descr_avg_1_gram_tf', 'query_descr_avg_2_gram_tf']]
    
    df_relevance = df_all[['id', 'relevance']]
    
    df_features.to_csv('data/train_features.csv')
    df_features.to_csv('data/train_relevance.csv')

        
        
        
            
        
    
        
        
    
    
    
