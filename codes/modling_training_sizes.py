import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import np_utils

author = 'Hassan'
date= 'May 13, 2018'
email= 'halhuzali@gmail.com'

def f_score(model, x_train, y_train, x_test, y_test, n):
    """

    :param model: classifier like logistic regression
    :param x_train: a matrix of vectors per each data-point
    :param y_train: an array of labels
    :param x_test: a matrix of vectors per each data-point
    :param y_test: an array of labels
    :param n: int(size of training data)
    :return: an F-score metric
    """
    # ToDo:convert labels to matrix as one-hot encoding
    y_train = np_utils.to_categorical(np.array(y_train))
    Y_test = np_utils.to_categorical(np.array(y_test))
    model.fit(x_train[:n], y_train[:n], nb_epoch=2, validation_data=(x_test, Y_test), verbose=2)
    #cls= model.fit(x_train[:n], y_train[:n])
    probs= model.predict(x_test)
    pred= probs.argmax(axis=-1)
    return f1_score(y_test, pred, average='weighted')


def split_training_data(train_data, start, step):
    """

    :param train_data: a matrix of vectors per each data-point
    :param start: int()
    :param step: int()
    :return: a list of numbers where they'll be used for training mini-batches
    """
    train_sizes = [i for i in range(start, train_data.shape[0] + 1, step)]
    return train_sizes


def training_size(cls, X_train, y_train, X_test, y_test, start, step):
    """

    :param cls: dict of name and model
    :param x_train: a matrix of vectors per each data-point
    :param y_train: an array of labels
    :param x_test: a matrix of vectors per each data-point
    :param y_test: an array of labels
    :return: a pandas dataframe
    """
    table = []
    train_sizes = split_training_data(X_train, start, step)
    for name, model in cls.items():
        for n in train_sizes:
            table.append({'Model': name,
                          'f1-score': f_score(model, X_train, y_train, X_test, y_test, n),
                          'train_size': n})
    return pd.DataFrame(table)


def ploting(df):
    """

    :param df: pandas dataframe
    :return: figure
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 7))
    fig = sns.pointplot(x='train_size', y='f1-score', hue='Model',
                        data=df)
    fig.set(ylabel="F-score")
    fig.set(xlabel="#training examples")
    fig.set(title="Assesing various training sizes")
    fig.legend_.set_title('Models')
    return fig

