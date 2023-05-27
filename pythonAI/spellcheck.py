import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import json
from textblob import TextBlob

first15 = 0
firstT15 = 0

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
            preCheck=(category, review_json['reviewText'])
            tb_phraseTest = TextBlob(str(preCheck))
            postCheck = str(tb_phraseTest.correct())
            review = Review(category, postCheck)
            train_reviews.append(review)
            if first15 < 15:
                print(postCheck)
                first15 += 1

train_container = ReviewContainer(train_reviews)

#prepare test data
test_reviews = []
for file in os.listdir('./data/test'):
    category = file.strip('test_').split('.')[0]
    with open(f'./data/test/{file}') as f:
        for line in f:
            review_json = json.loads(line)
            preCheckTest=(category, review_json['reviewText'])
            tb_phraseTest = TextBlob(str(preCheckTest))
            postCheckTest = str(tb_phraseTest.correct())
            review = Review(category, postCheckTest)
            test_reviews.append(review)
            if firstT15 < 15:
                print(postCheckTest)
                firstT15 += 1
            
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