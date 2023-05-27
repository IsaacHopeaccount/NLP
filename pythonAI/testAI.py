import os
import numpy as np
import random
import spacy
import spacy_transformers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import re

#practicing different techniques
import json

class Category:
  BOOKS = "BOOKS"
  CLOTHING = "CLOTHING"

train_x = ["i love the book", "this is a great book", "the fit is great", "i love the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(train_x)

print(vectorizer.get_feature_names_out())

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

test_x = vectorizer.transform(['i love the books'])

clf_svm.predict(test_x)
print(clf_svm.predict(test_x))

nlp = spacy.load("en_core_web_md")
print(train_x)

docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = ["I went to the bank and wrote a check", "let me check that out"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors =  [x.vector for x in test_docs]

clf_svm_wv.predict(test_x_word_vectors)
print(clf_svm_wv.predict(test_x_word_vectors))

regexp = re.compile(r"\bread\b|\bstory\b|book")

phrases = ["I liked that story.", "the car treaded up the hill", "this hat is nice"]

matches = []
for phrase in phrases:
  if re.search(regexp, phrase):
    matches.append(phrase)

print(matches)

import nltk

#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

phrase = "reading the books"
words = word_tokenize(phrase)

stemmed_words = []
for word in words:
  stemmed_words.append(stemmer.stem(word))

" ".join(stemmed_words)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

phrase = "reading the books"
words = word_tokenize(phrase)

lemmatized_words = []
for word in words:
  lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))

" ".join(lemmatized_words)

print(lemmatized_words)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

phrase = "Here is an example sentence demonstrating the removal of stopwords"

words = word_tokenize(phrase)

stripped_phrase = []
for word in words:
  if word not in stop_words:
    stripped_phrase.append(word)

" ".join(stripped_phrase)

print(stripped_phrase)

from textblob import TextBlob

phrase = "the book was horrible"

tb_phrase = TextBlob(phrase)

tb_phrase.correct()

tb_phrase.tags

tb_phrase.sentiment

print(tb_phrase.sentiment)


import spacy
import torch

nlp = spacy.load("en_core_web_md")
doc = nlp("Here is some text to encode.")

class Category:
  BOOKS = "BOOKS"
  BANK = "BANK"

train_x = ["good characters and plot progression", "check out the book", "good story. would recommend", "novel recommendation", "need to make a deposit to the bank", "balance inquiry savings", "save money"]
train_y = [Category.BOOKS, Category.BOOKS, Category.BOOKS, Category.BOOKS, Category.BANK, Category.BANK, Category.BANK]




docs = [nlp(text) for text in train_x]
train_x_vectors = [doc.vector for doc in docs]
clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)

test_x = ["check this story out", "i need to make a deposit", "i want to save money"]
docs = [nlp(text) for text in test_x]
test_x_vectors = [doc.vector for doc in docs]

clf_svm.predict(test_x_vectors)
print(clf_svm.predict(test_x_vectors))
     