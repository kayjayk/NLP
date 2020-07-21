#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
from time import time
from pprint import pprint
import json
from copy import copy
import pandas as pd
from pylab import scatter, show, legend, xlabel, ylabel
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import normalize

import os

class Topic_modeling:
    def __init__(self, class_num, n_components):
        prep_save_path = '/home/data/00_nh_poc/03_kdi_prep_prv_text_10000_2/'
        raw_save_path = '/home/data/00_nh_poc/01_kdi_raw_prv_txt_10000/'

        label_table = pd.read_excel('./200721_topic_modeling_10_NMF_1.xlsx', sheet_name='작업')

        origin_txt_names = label_table[label_table['after'] == class_num]['file_name']
        origin_txt_names_wo_ext = list(origin_txt_names.apply(lambda x: x.split('.')[0]))

        data_samples = []

        for fn in prep_file_list:
            with open(os.path.join(prep_save_path, fn), 'r', encoding='utf-8') as f:
                prep_text = f.read()

            origin_name = fn.split('_')[0]
            if origin_name in origin_txt_names_wo_ext:
                data_samples.append(prep_text)

        doc_dic = {}
        for fn, prep_text in zip(prep_file_list, data_samples):
            raw_fn = fn.split('_prep_prv_text.txt')[0] + '.txt'
            doc_dic[raw_fn] = {}
            with open(os.path.join(raw_save_path, raw_fn), 'r', encoding='utf-8') as f:
                raw_text = f.read()
            doc_dic[raw_fn]['raw_text'] = raw_text
            doc_dic[raw_fn]['prep_text'] = prep_text


            doc_dic[raw_fn]['nmf_1_topic_num'] = 0
            doc_dic[raw_fn]['nmf_1_topic_key_word'] = 0

        n_samples = 3000
        n_features = 1000
        n_components = n_components
        n_top_words = 20    # 각 topic 별 top 20 words

        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       #max_features=n_features,
                                       stop_words='korean')
        tfidf = tfidf_vectorizer.fit_transform(data_samples)
        nmf_1 = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        nmf_1_topic_set = print_top_words(nmf_1, tfidf_feature_names, n_top_words)
        doc_topic_nmf_1 = nmf_1.transform(tfidf)

        i = 0
        for raw_fn in doc_dic:
            nmf_1_topic_num = doc_topic_nmf_1[i].argmax()
            doc_dic[raw_fn]['nmf_1_topic_num'] = nmf_1_topic_num
            doc_dic[raw_fn]['nmf_1_topic_key_word'] = ' '.join(nmf_1_topic_set[nmf_1_topic_num])
            i+=1

        fn_list = []
        raw_text_list = []
        prep_text_list = []
        nmf_1_topic_num_list = []
        nmf_1_topic_key_word_list = []

        for raw_fn in doc_dic:
            fn_list.append(raw_fn)
            raw_text_list.append(doc_dic[raw_fn]['raw_text'])
            prep_text_list.append(doc_dic[raw_fn]['prep_text'])

            nmf_1_topic_num_list.append(doc_dic[raw_fn]['nmf_1_topic_num'])
            nmf_1_topic_key_word_list.append(doc_dic[raw_fn]['nmf_1_topic_key_word'])


        dic_df = {'file_name' : fn_list, 
                  'raw_text': raw_text_list, 
                  'prep_text' : prep_text_list, 
                  'nmf_1_topic' : nmf_1_topic_num_list, 
                  'nmf_1_key_word' : nmf_1_topic_key_word_list
                 }

        df = pd.DataFrame(dic_df) 

        excel_output_path = '/home/workspace/jaeyeun/01_nh_poc/04_kdi_prv_text_nmf_lda_round2'

        excel_file_name = 'topic_modeling_{}.xlsx'.format(n_components)

        df.to_excel(os.path.join(excel_output_path, excel_file_name))


    def print_top_words(model, feature_names, n_top_words):
        topic_set = []
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            topic_set.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()
        return topic_set

