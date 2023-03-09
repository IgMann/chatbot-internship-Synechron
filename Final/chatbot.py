# Importing libraries
import copy
import re
import time
import json
import itertools
import pandas as pd
import numpy as np
import copy as cp
import spacy
from array import array
import gensim.downloader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.neighbors import NearestNeighbors
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, Lambda
import keras.backend as K

class FAQ_chatbot():

    def __init__(self):
        self.__word2vec = gensim.downloader.load('glove-twitter-25')

        # self.__nlp = spacy.load("en_core_web_sm")

        qa_base = pd.read_csv('./insurance_qna_dataset.csv', sep='\t', index_col=0)
        qa_base.drop_duplicates(subset="Question", keep="first", inplace=True)
        self.__qa_base = qa_base

        file = open("qa_base_vectorized.json", "r")
        qa_base_vectorized = json.load(file)
        file.close()

        self.__n_neighbors = 50
        self.__nbrs_euclidean = NearestNeighbors(n_neighbors=self.__n_neighbors, metric="euclidean").fit(qa_base_vectorized)

        file = open("vocabulary.json", "r")
        self.__vocabulary = json.load(file)
        file.close()

    def tokenizer(self, text):
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

    def vectorizer(self, sentence):
        vectorized_sentence = []
        for i in range(len(sentence)):
            try:
                word = sentence[i]
                word_vectorized = self.__word2vec[word].tolist()
                word_spacy = self.__nlp(word)
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

    def token_mean(self, sentence):
        vector_length = len(sentence[0])
        mean_list = []
        for i in range(vector_length):
            local_mean_list = []
            for word in sentence:
                local_mean_list.append(word[i])
            mean_list.append(np.mean(local_mean_list))

        return mean_list

    def high_recall(self, question):
        question_tokenized = self.tokenizer(question)
        question_vectorized = self.vectorizer(question_tokenized)
        question_mean = self.token_mean(question_vectorized)
        distances, indices = self.__nbrs_euclidean.kneighbors(np.reshape(question_mean, (1, -1)))
        nbrs_list = self.__qa_base["Question"].iloc[indices[0]].tolist()

        return nbrs_list

    def high_precision(self, question, questions):

        embedding_questions = copy.deepcopy(questions)
        embedding_questions.append(question)

        vocabulary = self.__vocabulary
        q2n_questions = []

        for i in range(self.__n_neighbors + 1):
            q2n = []
            for word in self.tokenizer(embedding_questions[i]):
                if word not in vocabulary:
                    q2n.append(vocabulary[0])
                else:
                    q2n.append(vocabulary[word])

            q2n_questions.append(q2n)

        max_seq_length = max(map(len, q2n_questions))

        input_question = [q2n_questions[-1]]
        del q2n_questions[-1]
        ranked_questions = q2n_questions

        input_question = pad_sequences(input_question, maxlen=max_seq_length, padding='post')[0]
        ranked_questions = pad_sequences(ranked_questions, maxlen=max_seq_length, padding='post')

        def exponent_neg_manhattan_distance(left, right):
            ''' Helper function for the similarity estimate of the GRUs outputs'''
            return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

        # The visible layer
        left_input = Input(shape=(max_seq_length,), dtype='int32')
        right_input = Input(shape=(max_seq_length,), dtype='int32')

        embedding_dim = 25
        embedding_layer = Embedding(len(vocabulary), embedding_dim, input_length=max_seq_length, trainable=False)

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same GRU
        n_hidden = 50
        shared_gru = GRU(n_hidden)

        left_output = shared_gru(encoded_left)
        right_output = shared_gru(encoded_right)

        # Calculates the distance as defined by the MaGRU model
        magru_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        magru = Model([left_input, right_input], [magru_distance])

        magru.load_weights(filepath="checkpoint")

        input_question_list = [input_question.tolist() for i in range(self.__n_neighbors)]
        ranked_questions = ranked_questions.tolist()

        input_question_array = np.array(input_question_list)
        ranked_questions_array = np.array(ranked_questions)

        prediction = magru.predict([input_question_array, ranked_questions_array])
        question_similarity_list = list(itertools.chain(*prediction))

        index = question_similarity_list.index(max(question_similarity_list))

        question = questions[index]

        return question

    def get_answer(self, question):
        nearest_questions = self.high_recall(question)
        nearest_question = self.high_precision(question, nearest_questions)

        question_row = self.__qa_base.loc[self.__qa_base["Question"] == nearest_question]
        answer = question_row["Answer"].values[0]

        return answer

def main():
    question = "How Many People Live Without Health Insurance? "

    line = 100*'-'

    start = time.time()
    chatbot = FAQ_chatbot()
    checkpoint = time.time()

    answer = chatbot.get_answer(question)
    end = time.time()

    print(line)
    print("Question is: \n" + question)
    print("Answer is: \n" + answer)
    print(line)

    print("Initialisation time:", round((checkpoint-start), 2), 's')
    print("Execution time:", round((end-checkpoint), 2), 's')
    print(line)

if __name__ == '__main__':
    main()
