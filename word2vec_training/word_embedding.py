import nltk
#nltk.download('punkt')
print("all good")



#************************************************************************************************
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from pandas_profiling import ProfileReport
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import re
import errno
import json
import pickle
import glob
import multiprocessing
from time import time  # To time our operations

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.tokenize import word_tokenize
#from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
#from pylab import rcParams
#rcParams['figure.figsize'] = 15, 10

print('starting execution')

#************************************************************************************************



total_dataframe = pd.DataFrame()
#models = glob.glob('/kaggle/working/' + "*.pickle")
models = glob.glob('biorxiv_medrxiv.pickle') #for production comment this and uncomment above line
# print('models via glob===>'+str(models))
for individual_model in models:
    print(individual_model)
    # open a file, where you stored the pickled data
    file = open(individual_model, 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    print('Showing the pickled data:')
    my_df = pd.DataFrame(data)
    print(my_df.shape)
    my_df = my_df.replace(r'^\s*$', np.nan, regex=True)
    total_dataframe = total_dataframe.append(my_df, ignore_index=True)
    print(my_df.isnull().sum())

#************************************************************************************************

print('Total data in dataframe')
total_dataframe = total_dataframe.drop(['abstract_tripples', 'text_tripples','entities','key_phrases'], axis=1)
print(total_dataframe.shape)
print(total_dataframe.head())

#************************************************************************************************
total_dataframe.isnull().sum()
#************************************************************************************************
total_dataframe.dtypes
#************************************************************************************************
tf_df = pd.DataFrame()
tf_df['merged_text'] = total_dataframe['title'].astype(str) +  total_dataframe['abstract'].astype(str) +  total_dataframe['text'].astype(str)
tf_df['paper_id'] = total_dataframe['paper_id']
print(tf_df.head())

#************************************************************************************************
### This is word2vec training code we train it externaly and use the trained models

#sent = [row.split() for row in tf_df['merged_text']]

tf_df['tokenized_text'] = tf_df.merged_text.apply(lambda row: word_tokenize(row))
sent = tf_df.tokenized_text.to_list()
print(f"len(sent): {len(sent)}")
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
print(f"cores: {cores}")

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.save("word2vec.model")

#************************************************************************************************
### Using trained model
model = Word2Vec.load("word2vec.model")
vector = model.wv['disease']
print(vector)

#************************************************************************************************
print(vector.shape)