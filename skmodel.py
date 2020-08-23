import numpy as np
import pandas as pd
import matplotlib as plt
import nltk
nltk.download('punkt')
import re

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("training_data.csv")
print(data.shape)
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data.Text, data.Sentiment, test_size=0.2, random_state=0)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

X_train = [str(x) for x in X_train]

max_words = 5000
vectorizer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize, max_features=max_words)
x_train_counts = vectorizer.fit_transform(X_train)

tfmer = TfidfTransformer()
tf_x = tfmer.fit_transform(x_train_counts)
y_train_counts = vectorizer.transform(y_train)
tf_y = tfmer.transform(y_train_counts)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(tf_x, y_train)

predictions = clf.predict(tf_y)
print(sklearn.metrics.accuracy_score(y_test, predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))