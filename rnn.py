import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
from random import randint
wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist()
wordindex = wordsList.index("awesome")

wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')

wordindex = wordsList.index("awesome")
vec = wordVectors[wordindex]
print(vec)
numWords = []
f_in = open("posReview.csv",'r')
for line in f_in.readlines():
    counter = len(line.split())
    numWords.append(counter)
f_in = open("negReview.csv",'r')
for line in f_in.readlines():
    counter = len(line.split())
    numWords.append(counter)
totalReviews = len(numWords)
print('The total number of files is', totalReviews)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()
maxWords = 250

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
idMatrix = np.zeros((totalReviews, maxWords), dtype='int32')
fileCounter = 0
f_in = open("posReview.csv",'r')
for line in f_in.readlines():
        index = 0
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                idMatrix[fileCounter][index] = wordsList.index(word)
            except ValueError:
                idMatrix[fileCounter][index] = 999999
            index = index + 1
            if index >= maxWords:
                break
        fileCounter = fileCounter + 1

#idMatrix = np.zeros((totalReviews, maxWords), dtype='int32')
f_in1 = open("negReview.csv",'r')
for line in f_in1.readlines():
        index = 0
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                idMatrix[fileCounter][index] = wordsList.index(word)
            except ValueError:
                idMatrix[fileCounter][index] = 999999
            index = index + 1
            if index >= maxWords:
                break
        fileCounter = fileCounter + 1

np.save('idsMatrixAAAA', idMatrix)
idMatrix = np.load('idsMatrixAAAA.npy')
def trainingdata():
    labels = []
    reviewarrray = np.zeros([batchSize, maxWords])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        reviewarrray[i] = idMatrix[num-1:num]
    return reviewarrray, labels

def testingdata():
    labels = []
    reviewarrray = np.zeros([batchSize, maxWords])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        reviewarrray[i] = idMatrix[num-1:num]
    return reviewarrray, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxWords])
data = tf.Variable(tf.zeros([batchSize, maxWords, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
#Next Batch of reviews
    nextBatch, nextBatchLabels = trainingdata();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    #Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = testingdata();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)


def getSentenceMatrix(sentence):
    reviewarrray = np.zeros([batchSize, maxWords])
    sentenceMatrix = np.zeros([batchSize,maxWords], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for index,word in enumerate(split):
        try:
            sentenceMatrix[0,index] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,index] = 999999 #Vector for unkown words
    return sentenceMatrix


inputText = "That movie was terrible."
inputMatrix = getSentenceMatrix(inputText)


predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# predictedSentiment[0] represents output score for positive sentiment
# predictedSentiment[1] represents output score for negative sentiment

if (predictedSentiment[0] > predictedSentiment[1]):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")




secondInputText = "That movie was the best one I have ever seen."
secondInputMatrix = getSentenceMatrix(secondInputText)



predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
if (predictedSentiment[0] > predictedSentiment[1]):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")
