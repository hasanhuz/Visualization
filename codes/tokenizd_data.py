#!/usr/bin/env python
# -*- coding: utf-8 -*-

author = 'Hassan'
date= 'May 13, 2018'
email= 'halhuzali@gmail.com'

##module load hdf5-mpi
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import numpy as np

def BOW(data, num_features=80000):
    """

    :param data: list of tweets
    :param num_features: int()
    :return: a dict of top freq words in the data
    """
    word_space = Counter()
    for twt in data:
        for word in twt.split():
            word_space[word] = len(word_space)
    most_common = word_space.most_common(num_features)
    top_features = {w: c for w, c in most_common}
    return top_features

def tokenizeTweets(data, vocab):
    """

    :param data: list of tweets
    :param vocab: dict of words and their freq
    :return: array(data.indices), vocab_size(int)
    """
    tokenized_data = []
    for tweet in data:
        indices = [vocab.get(w, 0) for w in tweet.split(' ')]
        tokenized_data.append(indices)
    return np.array(tokenized_data), len(vocab)

def tokenizeData(X_train, X_test, vocab_size=10000):
    "tokenize data"
    # init tokenizer
    tokenizer = Tokenizer(nb_words=vocab_size, filters='\t\n', char_level=False)
    # use tokenizer to split vocab and index them
    #     vocab_size = len(tokenizer.nb_words) + 1
    tokenizer.fit_on_texts(X_train)
    vocab_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    return X_train, X_test, vocab_size, vocab_index

def paddingSequence(X_train, X_test, maxLen=30):
    """pad a sequence of data specified by a given  max_length

    list, int -> array"""
    #######equalize list of seq
    X_train = pad_sequences(X_train, maxLen, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxLen, padding='post', truncating='post')
    return X_train, X_test

def tokenizechar(X_train, X_test):
    "tokenize data"
    # init tokenizer
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    # use tokenizer to split vocab and index them
    char_index = tokenizer.word_counts
    char_size= len(char_index.keys())
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    return X_train, X_test, char_index, char_size


def tokenizeChars(data, chars):
    """return chars' indices

    list, dict -> dict, int"""
    tokenized_data = []
    char2idx = dict(zip(list(chars.decode('utf-8')), range(2, len(chars) + 2)))
    for tweet in data:
        indices = [char2idx.get(ch, 1) for ch in tweet]
        tokenized_data.append(indices)
    return np.array(tokenized_data), len(char2idx), char2idx
