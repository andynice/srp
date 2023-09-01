import nltk

import numpy as np
import pandas as pd

import os
import re
import errno
import json
import pickle
import glob
import multiprocessing
from time import time

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.tokenize import word_tokenize

print('starting execution')

#************************************************************************************************

filename = "./data/en_output.csv"
tf_df = pd.read_csv(filename)

#************************************************************************************************

print('Total data in dataframe')
print(tf_df.shape)
print(tf_df.head())

#************************************************************************************************
tf_df.isnull().sum()
#************************************************************************************************
tf_df.dtypes
#************************************************************************************************
### This is word2vec training code we train it externaly and use the trained models

#sent = [row.split() for row in tf_df['merged_text']]

tf_df['tokenized_text'] = tf_df['clean_tweets'].apply(lambda row: word_tokenize(row))
sent = tf_df.tokenized_text.to_list()
# print(f"sent: {sent}")
print(f"len(sent): {len(sent)}")
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
print(f"cores: {cores}")


# w2v_model = Word2Vec(min_count=20,
w2v_model = Word2Vec(min_count=1,
                     window=2,
                     vector_size=500,
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
vector = model.wv['love']
print(vector)

#************************************************************************************************
print(vector.shape)