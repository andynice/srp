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
import datetime
from time import time

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.tokenize import word_tokenize

print('starting execution')

#************************************************************************************************
total_dataframe = pd.DataFrame()
date_ranges = [['2021-01-01', '2021-02-01'], ['2021-02-01', '2021-03-01']]
for date_range in date_ranges:
    start = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    for date in date_generated:
        date_str = date.strftime("%Y-%m-%d")

        filename = f"./data/en_{date_str}_output.csv"
        df_data = pd.read_csv(filename)
        print(f"df_data.shape: {df_data.shape}")
        df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
        total_dataframe = total_dataframe.append(df_data, ignore_index=True)
        print(df_data.isnull().sum())

#************************************************************************************************

print('Total data in dataframe')
print(f'total_dataframe.shape: {total_dataframe.shape}')
print(f'total_dataframe.head(): {total_dataframe.head()}')
# output_file = "./output/test_output.csv"
# total_dataframe.to_csv(output_file, mode='a', index=False, header=True)

#************************************************************************************************
print(f"total_dataframe.isnull().sum(): {total_dataframe.isnull().sum()}")
#************************************************************************************************
print(f"total_dataframe.dtypes: {total_dataframe.dtypes}")
#************************************************************************************************
## This is word2vec training code we train it externaly and use the trained models

# sent = [row.split() for row in total_dataframe['merged_text']]

total_dataframe['tokenized_text'] = total_dataframe['clean_tweets'].apply(lambda row: word_tokenize(row))
sent = total_dataframe.tokenized_text.to_list()
# print(f"sent: {sent}")
print(f"len(sent): {len(sent)}")
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
print(f"cores: {cores}")


# w2v_model = Word2Vec(min_count=20,
w2v_model = Word2Vec(min_count=20,
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