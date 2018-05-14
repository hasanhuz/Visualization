from modling_training_sizes import training_size, ploting
from loading_data import loadingData
from tokenizd_data import tokenizeData, paddingSequence
from lstm_model import LSTM_model
from sys import argv

author = 'Hassan'
date= 'May 13, 2018'
email= 'halhuzali@gmail.com'

########################
#usage: run using command line
    #python model_confg.py train_file.csv test_file.csv picture_name
#######################
#ToDo: loading data
X_train, y_train = loadingData(argv[1])
X_test, y_test = loadingData(argv[2])

#ToDO: tokenizing data
X_train, X_test, vocab_size, vocab_index = tokenizeData(X_train, X_test)
X_train, X_test, = paddingSequence(X_train, X_test)

#ToDO: pass model after compling #models= #a dict(name, model) ex. {'LSTM':lstm(),...}
model= LSTM_model(vocab_size)
models= {'LSTM': model}

#ToDo: pass int()
start= 2000 # int(): where it'll be selected for training first batch
step = 2000 # int(): where it'll be incremented for following batches

#Calling functions #nb_epochs,
df = training_size(models, X_train, y_train, X_test, y_test, start, step)
print(df)
fig= ploting(df)

#ToDo:saving figure given a filename
filename=argv[3]
fig.figure.savefig(filename)
