# Importing libraries

import re
import time
import json
import statistics
import pandas as pd
import numpy as np
import copy as cp
import spacy
import gensim.downloader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def tokenizer(text):
    ''' Preprocess and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def vectorizer(sentence):
    vectorized_sentence = []
    for i in range(len(sentence)):
        try:
            word = sentence[i]
            word_vectorized = word2vec[word].tolist()
            word_spacy = nlp(word)
            for token in word_spacy:
                pos = token.pos_
                break
            if pos == "NOUN" or pos == "PROPN" or pos == "ADJ":
                pos_coef = 2
            elif pos == "VERB":
                pos_coef = 1
            elif pos == "NUM":
                pos_coef = 0
            else:
                pos_coef = 0.5
            word_vectorized = [pos_coef * element for element in word_vectorized]
            vectorized_sentence.append(word_vectorized)
        except:
            pass

    return vectorized_sentence


def token_mean(sentence):
    vector_length = len(sentence[0])
    mean_list = []
    for i in range(vector_length):
        local_mean_list = []
        for word in sentence:
            local_mean_list.append(word[i])
        mean_list.append(np.mean(local_mean_list))

    return mean_list

word2vec = gensim.downloader.load('glove-twitter-25')
print("Word2vec model imported!!!")
nlp = spacy.load("en_core_web_sm")

qa_base = pd.read_csv('./insurance_qna_dataset.csv', sep='\t', index_col=0)
qa_base.drop_duplicates(subset="Question", keep="first", inplace=True)
questions_list = qa_base["Question"].tolist()
questions_number = qa_base.shape[0]

qa_base_vectorized_mean = []

for i in range(questions_number):
    print(str(i+1) + ". iteration")
    sentence = questions_list[i]
    token_words = tokenizer(sentence)
    sentence_vectorized = vectorizer(token_words)
    sentence_mean = token_mean(sentence_vectorized)
    qa_base_vectorized_mean.append(sentence_mean)

print("Vectorization of base finished!!!")

out_file = open("qa_base_vectorized.json", "w")
json.dump(qa_base_vectorized_mean, out_file)
out_file.close()

