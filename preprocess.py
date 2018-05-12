#import lib
import pandas as pd
import xml.etree.ElementTree as ET
#import XML data from .Xml files
tree = ET.parse(r"C:\Users\aakar\Desktop\neg.xml")
tree1 = ET.parse(r"C:\Users\aakar\Desktop\positivecopy.xml")
root = tree.getroot()
root1 = tree1.getroot()

count  = 0
negative = []
negvalue = []
positive = []
posvalue = []
#making a list of reviews 
for app in root.findall('review'):
    for l in app.findall('review_text'):
        count+=1 
        print("%s" % (l.text))
        negative.append(l.text)
        negvalue.append(0)
for app1 in root1.findall('review'):
    for l1 in app1.findall('review_text'):
        count+=1 
        print("%s" % (l1.text))
        positive.append(l1.text)
        posvalue.append(1)

#elimanating extra symbols and converting to lower case
negstrip= []
posstrip = []
for i in range(len(negative)):
    string=negative[i].strip()
    string.strip("\n")
    negstrip.append(string.lower().replace("\n", " "))
    print(negstrip[i])
for i in range(len(positive)):
    string1=positive[i].strip()
    string1.strip("\n")
    posstrip.append(string1.lower().replace("\n", ""))
    print(posstrip[i])
#making dataframe 
data_tuples = list(zip(negstrip,negvalue))
negdata=pd.DataFrame(data_tuples, columns=['Review','label'])
data_tuples1 = list(zip(posstrip,posvalue))
posdata=pd.DataFrame(data_tuples1, columns=['Review','label'])
#making .csv files 
negdata.to_csv(r'C:\Users\aakar\Desktop\negReview.csv', sep=',')
posdata.to_csv(r'C:\Users\aakar\Desktop\posReview.csv', sep=',')
