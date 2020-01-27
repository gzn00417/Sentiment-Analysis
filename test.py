import numpy
from nltk.classify import NaiveBayesClassifier
import linecache
import random

pos_text=open('data\\rt-polarity.pos','r')
neg_text=open('data\\rt-polarity.neg','r')

Sum_Line=5000
Devide_Part=1
training_data=[]
dict_num={}

def line_clean(line):
    words=line.lower().strip().split()
    for word in words:
        if word in ',.?!;:':
            words.remove(word)
    return ' '.join(words)

#IDF预处理
for i in range(int(Sum_Line*Devide_Part)):
    new_line=[word for word in line_clean(linecache.getline('data\\rt-polarity.pos',i)).split()]
    for word in new_line:
        if word in dict_num.keys():
            dict_num[word]+=1.0
        else:
            dict_num[word]=1.0
for i in range(int(Sum_Line*Devide_Part)):
    new_line=[word for word in line_clean(linecache.getline('data\\rt-polarity.neg',i)).split()]
    for word in new_line:
        if word in dict_num.keys():
            dict_num[word]+=1.0
        else:
            dict_num[word]=1.0
max_num=0.0
for word in dict_num.keys():
    max_num=max(max_num,dict_num[word])
for word in dict_num.keys():
    dict_num[word]=float(1-dict_num[word]/max_num)


pro_lim1=0.01
pro_lim2=0.02

#特殊特征标记
def preprocess1(s):
    return {word : (dict_num[word]>pro_lim1) for word in line_clean(s).split()}
def preprocess2(s):
    return {word : (word in dict_num.keys() and dict_num[word]>pro_lim2) for word in line_clean(s).split()}

for i in range(int(Sum_Line*Devide_Part)):
    training_data.append([preprocess1(line_clean(linecache.getline('data\\rt-polarity.pos',i))),1])

for i in range(int(Sum_Line*Devide_Part)):
    training_data.append([preprocess1(line_clean(linecache.getline('data\\rt-polarity.neg',i))),-1])

model = NaiveBayesClassifier.train(training_data)

test_text=open("data\\rt-polarity.test","r")
for line in test_text:
    probdist = model.prob_classify(preprocess2(line_clean(line)))
    print probdist.max()