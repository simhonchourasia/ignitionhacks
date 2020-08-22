# Import statements
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get data from csv file (in same folder location)
train_set = pd.read_csv("trainset.csv")
dev_set = pd.read_csv("devset.csv")
'''
print(train_set.shape)
print(train_set.head())
print(dev_set.shape)
print(dev_set.head())
'''

# Split the datasets
X_train = train_set.Text
y_train = train_set.Sentiment
X_dev = dev_set.Text
y_dev = dev_set.Sentiment

# Print out basic information about the dataset
print(X_train.head())
print(y_train.head())
print(X_dev.head())
print(y_dev.head())

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 5000
oov_token = "<OOV>"
max_len = 280

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

train_seqs = tokenizer.texts_to_sequences(X_train)
dev_seqs = tokenizer.texts_to_sequences(X_dev)
train_padded = pad_sequences(train_seqs, maxlen=max_len, truncating="post", padding="post")
dev_padded = pad_sequences(dev_seqs, maxlen=max_len)
'''
# MODEL: default
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
'''

#'''
# MODEL: CNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_len),
    tf.keras.layers.Conv1D(16, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
model.summary()
#'''

'''
# MODEL: BIDIRECTIONAL LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
'''



epochs = 10
history = model.fit(train_padded, y_train, epochs=epochs, validation_data=(dev_padded, y_dev))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["accuracy", "val_accuracy"])
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["loss", "val_loss"])
plt.show()


def predict_sentence(model, sentence):
    sample_seq = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sample_seq, padding='post', maxlen=max_len)
    classes = model.predict(padded)
    for x in range(len(padded)):
        print(sentence[x])
        print(classes[x])
        print("----------")


sentences = ["dap me up bro", "i hate you and ur mum", "this is why she left you", "thanks for the gold kind stranger"]
predict_sentence(model, sentences)