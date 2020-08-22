import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing
import tensorflow_hub as hub
import matplotlib as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data.csv', usecols=['Text', 'Sentiment'])
n = 850000
d = 256
ep = 20
batch = 500
maxlen = 280

# SPLIT DATA
msk = np.random.rand(len(df)) < 0.85
trainset = df[msk]
testset = df[~msk]

# LOAD UNIVERAL SERIAL ENCODER
embed = tf.saved_model.load('tmp/tfhub_modules/use_module')
trainlist = trainset['Text'].tolist()
splitlist = np.array_split(trainlist,100)
# GET EMBEDDINGS
embedlist=[]
for i in splitlist:
     embedlist.append(embed(i))
print(embedlist)
#
# ytrain=trainset['Sentiment'].values
# ytest=testset['Sentiment'].values


# model = models.Sequential()
# model.add(layers.Dense(128, activation = 'relu'))
# model.add(layers.Dense(2, activation = 'softmax'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.sumamry()
#
# history = model.fit(training_embeddings, ytrain, epochs=ep, validation_split=0.1, shuffle=True, batch_size=batch)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
