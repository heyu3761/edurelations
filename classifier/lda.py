

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import string
import numpy as np
import math
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords as pw
import pandas as pd


def textPrecessing(text):
    text = text.lower()
    for c in string.punctuation:
        text = text.replace(c, ' ')
    wordLst = nltk.word_tokenize(text)
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]

    return " ".join(filtered)

def print_top_words(model, feature_names, n_top_words):
    
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print(model.components_)
    a = model.components_
    print(a.shape)
   
    
def kl(x,y):
    k_x = set(x)
    p = []
    for i in k_x:
        p.append(x.count(i) / len(x))
    
    k_y = set(y)
    q = []
    for i in k_y:
        q.append(y.count(i) / len(y))
    
    KL = 0.0
    print(p,q)
    for i in range(len(k_x)):
        KL += p[i] * math.log(p[i] / q[i], 2)
    return KL
    
    
def ldaEntropy(lda):
    cnt = len(lda)
    ans = np.zeros((cnt, cnt))
    pair = np.zeros((cnt*cnt,5))
    for i in range(cnt):
        for j in range(cnt):
            ret = -np.sum(lda[i] * np.log(lda[i]))
            ans[i][j] = ret
            pair[j * cnt + i][0] = i
            pair[j * cnt + i][1] = j
            pair[j * cnt + i][2] = ret
            ret1 = -np.sum(lda[j] * np.log(lda[j]))
            pair[j * cnt + i][3] = ret1
            pair[j * cnt + i][4] = ret + kl(list(lda[i]),list(lda[j]))
    df=DataFrame(ans)
    df.to_excel('../../canusedata/matrix/feature/lda.xlsx')
    df = DataFrame(pair)
    df.to_excel('../../canusedata/matrix/feature/pair_lda.xlsx')
    
    
concepts = []
conceptDic = {}

index = 0
for line in open("../../canusedata/matrix/concept.txt", encoding="utf-8"):
    concept = line.strip("\t\n")
    concepts.append(concept)
    conceptDic[concept] = index
    index = index + 1
    

conceptMat = [[0]*index]*index 


pre_matrix = pd.read_csv("../../canusedata/matrix/matrix_with_annotators_names.csv")
leni, lenj = pre_matrix.shape
pre_matrix = pre_matrix.values
for i in range(leni):
    for j in range(1,lenj):      
        if str(pre_matrix[i][j])!='0' :
            conceptMat[i][j-1] = 1
            conceptMat[j-1][i] = -1



summaryDoc = []
contentDoc = []
titleDoc = []
linksDoc = []
cataDoc = []
WikiFile = "../../canusedata/matrix/wiki/"

for concept in concepts:
    if concept == "CLIENT/SERVER MODEL":
        concept = "client_server model"
    if concept == "TRANSMISSION CONTROL PROTOCOL/INTERNET PROTOCOL NETWORK":
        concept = "TRANSMISSION CONTROL PROTOCOL_INTERNET PROTOCOL NETWORK"  
        
    filename = WikiFile + concept + "_content.txt"
    f = open(filename, encoding="utf-8")
    doc = f.read()
    doc = doc.lower()
    contentDoc.append(doc)
    
    filename = WikiFile + concept + "_summary.txt"
    f = open(filename, encoding="utf-8")
    doc = f.read()
    doc = doc.lower()
    summaryDoc.append(doc)
    
    filename = WikiFile + concept + "_title.txt"
    f = open(filename, encoding="utf-8")
    doc = f.read()
    doc = doc.lower()
    titleDoc.append(doc)
    
    filename = WikiFile + concept + "_links.txt"
    f = open(filename, encoding="utf-8")
    doc = f.read()
    doc = doc.lower()
    linksDoc.append(doc)
    
    filename = WikiFile + concept + "_categories.txt"
    f = open(filename, encoding="utf-8")
    doc = f.read()
    doc = doc.lower()
    cataDoc.append(doc)


docLst = contentDoc
print(len(docLst))
'''
for desc in data_samples :
    docLst.append(textPrecessing(desc).encode('utf-8'))
'''
        
stopWords = pw.words("english")
n_features = 2500

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,stop_words=stopWords)
tf = tf_vectorizer.fit_transform(docLst)


n_topic = 30
lda = LatentDirichletAllocation(n_components=5,
                                max_iter=1000,
                                learning_method='batch',
                                verbose=True)
ans = lda.fit(tf)
docres = lda.fit_transform(tf)       
print(docres)
ldaEntropy(docres)

n_top_words=20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)