# Sentiment-Analysis-of-Reviews-using-Machine-Learning-algorithms-on-Textual-data
CSCI 59000 BIG DATA ANALYTICS PROJECT

## Description

* Programmed a XML parser in python using xml.etree.ElementTree package
* Text data is Pre-processed by removing special characters 
* Word embeddings of text are created using Word2Vec tool and tokenized.
* A deep learning model is created using TensorFlow framework by implementing Long-short term memory (LSTM) based Recurrent neural networks.
* Bigrams are created for the text after undergoing pre-processing, which includes removing stop words and stemming.
* Naïve-bayes classification model is built using Bigrams and nltk package.
* Performance analysis of both models is done by drawing ROC curves, by comparing accuracies, and Area Under Curve.

### Requirements
**Python packages:** numpy, tensorflow, matplotlib, nltk, sklearn, itertools

## Dataset

**sample Amazon XML dataset**

<img src="https://github.com/Premchand95/Sentiment-Analysis-of-Reviews-using-Machine-Learning-algorithms-on-Textual-data/blob/master/img/Picture1.jpg" height="350" width="550">

## LSTM RNN tensorflow model

**used tensorflow**

<img src="https://github.com/Premchand95/Sentiment-Analysis-of-Reviews-using-Machine-Learning-algorithms-on-Textual-data/blob/master/img/Picture2.png">

## Bigrams Naive-bayes model

**used nltk**

<img src="https://github.com/Premchand95/Sentiment-Analysis-of-Reviews-using-Machine-Learning-algorithms-on-Textual-data/blob/master/img/Picture3.jpg" height="350" width="350">

## Results

Predictive Model | Accuracy
------------ | ------------- 
Naive Bayes Classification | 68.5
RNN using LSTM | 70.83
Naive Bayes Classification with Bigrams | 74.4

**Naive Bayes Classification with Bigrams showed higest acuracy using nltk**
