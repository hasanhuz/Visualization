#!/usr/bin/env python
# -*- coding: utf-8 -*-

author = 'Hassan'
date= 'May 13, 2018'
email= 'halhuzali@gmail.com'

from sys import argv
import pandas as pd
import numpy as np
import re
import csv


def tokenizedTweet(tweet):
    """

    :param tweet: str()
    :return: a tokenized tweet
    """
    from tweetokenize import Tokenizer
    tokens = Tokenizer(lowercase=True, normalize=2, allcapskeep=False, ignorestopwords=True)
    tokenize= tokens.tokenize(tweet)
    return ' '.join(tokenize)

def loadingData(filename):
    """return a list of tweets and their labels

    str(filename) -> array, array"""
    print('...Loading Data...')
    df = pd.read_csv(filename, iterator=True, chunksize=100000, sep='","', header=0, engine='python')  # error_bad_lines=False,
    df = pd.concat(df, ignore_index=True)
    df.columns = ['label', 'tweet']
    print('...Cleaning Data...')
    # filter non-Arabic chars, convert @name to "<USER>", conver http to "<URL>", keep a tweet with more than 5 words at min
    df['tweet'].replace(to_replace="\s+", value=r" ", regex=True, inplace=True)
    # df = df[df.tweet.str.split().str.len() >= 5]
    #df['tweet']= df.apply(lambda row: tokenizedTweet(row['tweet']), axis=1)
    df['tweet'] = df.apply(lambda row: row['tweet'].lower().strip().strip('"'), axis=1)
    df['label'] = df.apply(lambda row: row['label'].lower().strip().strip('"'), axis=1)

    df = df.sample(frac=1)

    print('...Extracting tweets and labels as np.array...')
    # Convert tweets and labels to python lists
    df['label'] = df.apply(lambda row: row['label'].upper().strip(), axis=1)
    emo_label_map = {"ANGER": 0, "DISGUST": 1, "FEAR": 2, "JOY": 3, "SAD": 4, "SURPRISE": 5}
    df['label'] = df['label'].map(emo_label_map)
    list_tweets = np.array(df['tweet'])
    len(list_tweets)
    labels = np.array(df['label'])
    len(labels)
    print('...Done...')
    return list_tweets[:10000], labels[:10000]
