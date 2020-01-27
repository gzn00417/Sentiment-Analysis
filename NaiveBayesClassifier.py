import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import linecache
import random

pos_text=open('data\\rt-polarity.pos','r')
neg_text=open('data\\rt-polarity.neg','r')

Sum_Line=5000
Devide_Part=0.8
data=[]
dict_num={}

def line_clean(line):
    words=line.lower().strip().split()
    for word in words:
        if word in ',.;:':
            words.remove(word)
    return ' '.join(words)

def preprocess1(s):
    return {word : True for word in line_clean(s).split()}

for i in range(int(Sum_Line*Devide_Part)):
    data.append([preprocess1(line_clean(linecache.getline('data\\rt-polarity.pos',i))),'pos'])
    data.append([preprocess1(line_clean(linecache.getline('data\\rt-polarity.neg',i))),'neg'])

random.shuffle(data)
training_data=data[:int(Sum_Line*Devide_Part)]
test_data=data[int(Sum_Line*Devide_Part):]
model = nltk.NaiveBayesClassifier.train(training_data)
print nltk.classify.accuracy(model,test_data)

input_reviews = [
    "It is an amazing movie",
    "This is a dull movie. I would never recommend it to anyone.",
    "The cinematography is pretty great in this movie",
    "The direction was terrible and the story was all over the place"
]
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
print "\n预测:"
for review in input_reviews:
    print "\n评论:", review
    probdist = model.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print "预测情绪:", pred_sentiment
    print "可能性:", round(probdist.prob(pred_sentiment), 2)