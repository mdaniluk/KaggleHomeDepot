# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:16:03 2016

@author: dudu
"""

import pandas as pd
import re

from nltk.stem.porter import PorterStemmer


def load_train_and_desc(data_file, lucene_file):
    df_all = pd.read_csv(data_file)
    df_lucene_features = pd.read_csv(lucene_file, \
        names=['id', 'in_top_lucene', 'lucene_ranking_place', 'lucene_score'])
    df_pro_desc = pd.read_csv('data/product_descriptions.csv')
    df_attr = pd.read_csv('data/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"]\
    [["product_uid", "value"]].rename(columns={"value": "brand"})
    df_all = pd.merge(df_all, df_lucene_features, how='left', on='id')
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    
    df_db = pd.merge(df_pro_desc, df_brand, how='left', on='product_uid')
    print 'data loaded'
    return (df_all, df_db, df_attr)
 
 
       
def get_attributes_as_text(uid, df_attr):
    attr_list = []
    for _, a in df_attr[df_attr.product_uid == uid].iterrows():
        attr_list.append(' '.join([str(a['name']), str(a['value'])]))
    content = (', '.join(attr_list))
    return preprocess_text(content)
 
       
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

    
def safely_stem(word):
    stemmer = PorterStemmer()
    try:
        return stemmer.stem(word)
    except UnicodeDecodeError:
        return unicode(word, errors='ignore')
 
   
def str_stem(s): 
    strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,\
        'seven':7,'eight':8,'nine':0}
        
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        tmp1 = s.split(" ")
        tmp = [safely_stem(z) for z in tmp1]
        s = (" ").join(tmp)
        
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"
    
    
def preprocess_text(text):
    try:
        text = str(text)
        return str_stem(text)
    except Exception:
        return ''  
        

def create_features(data_file, lucene_file, features_file, add_relevance=False):
    df_all, df_db, df_attr = load_train_and_desc(data_file, lucene_file)
        
#    df_all = df_all[:100]
    df_all['lucene_ranking_place'] = \
        (201 - df_all['lucene_ranking_place']) / 200.0
    df_all['prepr_attrs'] = \
        df_all['product_uid'].map(lambda x: get_attributes_as_text(x, df_attr))
    df_all['prepr_descr'] = \
        df_all['product_description'].map(lambda x: preprocess_text(x))
    df_all['prepr_query'] = \
        df_all['search_term'].map(lambda x: preprocess_text(x))
    df_all['prepr_title'] = \
        df_all['product_title'].map(lambda x: preprocess_text(x))
    df_all['prepr_brand'] = \
        df_all['brand'].map(lambda x: \
        preprocess_text(x) if x != None else '')
    
    print 'Strings preprocessed. Calculating features...'    
    
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
    
    if (add_relevance):
        df_relevance = df_all[['id', 'relevance']]
        df_relevance.to_csv('data/relevance.csv', index=False)
    
    df_features.to_csv(features_file, index=False)
    
        
if __name__ == '__main__':
    
    create_features('data/train.csv', 'AllenLucene/data/lucene_train.csv', \
    'data/features_train.csv', True)
    create_features('data/test.csv', 'AllenLucene/data/lucene_test.csv', \
    'data/features_test.csv', False)
 
    
    

        