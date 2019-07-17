# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:52:00 2018

@author: Prem chand
"""
import pandas as pd
import collections
import nltk.classify.util
import nltk.metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#import re
import itertools
from nltk.classify import NaiveBayesClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import matplotlib.pyplot as plt

stop_words=set(stopwords.words('english'))
my_set=set(line.strip() for line in open('stopwords.txt'))

def labeltoint(label):
    if label == 'pos':
        return 1
    else:
        return 0
#remove symbols,Punctuation and break tags
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    string.replace('[^\w\s]','')
    return re.sub(strip_special_chars, "", string.lower())

def bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    words=words.split(" ")
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if not w in my_set]
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
classifier=None
def naiveBaseClassifier(featx):
    posdata=pd.read_csv(r'posReview1.csv', sep=',')
    negdata=pd.read_csv(r'negReview1.csv', sep=',')
    poslist=posdata["Review"].tolist()
    neglist=negdata["Review"].tolist()
    neglist=neglist[0:12248]+neglist[12250:]
    finalpos=[]
    finalneg=[]
    for item in poslist:
        item1=cleanSentences(item)
        finalpos.append(item1)
    for item in neglist:
        item1=cleanSentences(item)
        finalneg.append(item1)
    negfeats = [(featx(review),'neg') for review in finalneg]
    posfeats = [(featx(review), 'pos') for review in finalpos]

    negcutoff = int(len(negfeats)*0.5)
    poscutoff = int(len(posfeats)*0.5)

    traindata = negfeats[:negcutoff] + posfeats[:poscutoff]
    testdata = negfeats[negcutoff:] + posfeats[poscutoff:]

    classifier = NaiveBayesClassifier.train(traindata)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    y_true, y_score = [], []
    for i, (feats, label) in enumerate(testdata):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            y_true.append(labeltoint(label))
            y_score.append(labeltoint(observed))
            testsets[observed].add(i)

    print('accuracy:', nltk.classify.util.accuracy(classifier, testdata))
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC: %.2f" % pr_auc)
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print ("ROC AUC: %.2f" % roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    classifier.show_most_informative_features()

naiveBaseClassifier(bigrams)
