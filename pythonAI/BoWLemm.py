import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import json
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# define lemmatiser
def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)



class Review:
    def __init__(self, category, text):
        self.category = category
        self.text = text    
        
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
    
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_y(self):
        return [x.category for x in self.reviews]

#prepare training data
train_reviews = []
all_categories = []
for file in os.listdir('./data/training'):
    category = file.strip('train_').split('.')[0]
    all_categories.append(category) 
    with open(f'./data/training/{file}') as f:
        for line in f:
            review_json = json.loads(line)
            #review = Review(category, review_json['reviewText'])
            rawTrain=(review_json['reviewText'])
            tb_phrase = TextBlob(str(rawTrain))
            lemmTrain = str(tb_phrase.correct())
            lemmTrain = lemmatize_with_postag(lemmTrain)
            words=word_tokenize(lemmTrain)
            words=[w for w in words if not w in stop_words]
            lemm=' '.join(words)
            review = Review(category, lemmTrain)
            train_reviews.append(review)

train_container = ReviewContainer(train_reviews)

#prepare test data
test_reviews = []
for file in os.listdir('./data/test'):
    category = file.strip('test_').split('.')[0]
    with open(f'./data/test/{file}') as f:
        for line in f:
            review_json = json.loads(line)
            rawTest=(review_json['reviewText'])
            tb_phraseTest = TextBlob(str(rawTest))
            lemmTest = str(tb_phraseTest.correct())
            lemmTest = lemmatize_with_postag(lemmTest)
            words=word_tokenize(lemmTest)
            words=[w for w in words if not w in stop_words]
            lemmTest=' '.join(words)
            review = Review(category, lemmTest)
            test_reviews.append(review)
            
test_container = ReviewContainer(test_reviews)

corpus = train_container.get_text()
vectorizer = CountVectorizer(binary=True)
train_x = vectorizer.fit_transform(corpus) # training text converted to vector

clf = svm.SVC(kernel='linear')
clf.fit(train_x, train_container.get_y())

test_corpus = test_container.get_text()
test_x = vectorizer.transform(test_corpus)
print("Overall Accuracy:", clf.score(test_x, test_container.get_y()))

#predict
y_pred = clf.predict(test_x)

#print results
print("f1 scores by category")
print(all_categories)
print(f1_score(test_container.get_y(), y_pred, average=None, labels=all_categories))