from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Dropout

author = 'Hassan'
date= 'May 13, 2018'
email= 'halhuzali@gmail.com'

def LSTM_model(vocab_size):
    """

    :param pretrained_embedding:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param vocab_size:
    :return:
    """

    LSTM_DIM=64
    dim_size= 64
    word = Sequential()
    word.add(Embedding(input_dim=vocab_size, output_dim=dim_size, input_length=30))
    word.add(Dropout(0.5))
    word.add(LSTM(LSTM_DIM))
    word.add(Dense(6))
    word.add(Activation('softmax'))
    word.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    word.summary()
    return word
