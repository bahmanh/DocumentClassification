import pandas as pd 
import sklearn
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from plot_cm import plot_confusion_matrix
from plot_learning_curve import plot_learning_curve

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

import gensim, logging
from gensim.models import Word2Vec
from scipy import sparse

a_score = []
def loadData(filePath="./data/dataset.csv"):
    #data = pd.read_csv(filePath, header=0)
    data = pd.read_csv(open(filePath,'rU'), encoding='ISO-8859-1', engine='c')
    return data["Title"],data["Conference"]

def preProcessing(features):
    num_titles = features.size
    clean_wordlist = []
    clean_titles = []
    stops = set(stopwords.words('english'))
    for i in range( 0, num_titles):
        #letters_only = re.sub("[^a-zA-Z]", " ", features[i]) 
        words = features[i].lower().split()
        words = [w.lower() for w in words if not w in stops]  
        clean_wordlist.append(words)
        clean_titles.append(" ".join(words))
    return clean_titles, clean_wordlist

def getDTMByTFIDF(features,nfeatures):
    tfIdf_vectorizer = TfidfVectorizer(max_features=nfeatures)
    dtm = tfIdf_vectorizer.fit_transform(features).toarray()
    return dtm,tfIdf_vectorizer


def featuresByChiSq(features,labels,nFeature=5000):
    chi2_model = SelectKBest(chi2,k=nFeature)
    dtm = chi2_model.fit_transform(features,labels)
    return dtm,chi2_model

def featuresByInformationGain(features,labels):
    treeCL = tree.DecisionTreeClassifier(criterion="entropy")
    treeCL = treeCL.fit(features,labels)
    transformed_features = SelectFromModel(treeCL,prefit=True).transform(features)
    return transformed_features

def featuresByLSA(features,ncomponents=100):
    svd = TruncatedSVD(n_components=ncomponents)
    normalizer =  Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    dtm_lsa = lsa.fit_transform(features)
    return dtm_lsa
    
def makeFeatureVec(words, model, num_features):
    feature_vec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec,model[word]) 

    feature_vec = np.divide(feature_vec,nwords)
   
    return feature_vec

def getAvgFeatureVecs(title, model, num_features):
    counter = 0.
    titleFeatureVecs = np.zeros((len(title), num_features),dtype="float32")
    for t in title:
        titleFeatureVecs[int(counter)] = makeFeatureVec(t, model,num_features)
        counter = counter + 1.
    return titleFeatureVecs


def crossValidate(document_term_matrix,labels,classifier="SVM",nfold=2):
    clf = None
    precision = []
    recall = []
    fscore = []
    if classifier == "NN":
       clf = MLPClassifier(hidden_layer_sizes=(50), activation='relu', solver='sgd', alpha=1e-2, random_state=None)   
    elif classifier == "LR":
        clf = linear_model.LogisticRegression(C=1e3)
        #clf = tree.DecisionTreeClassifier()
    if classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "NB":
        clf = GaussianNB()
    elif classifier == "SVM":
        clf = LinearSVC()
    elif classifier == "KNN":
        clf = NearestCentroid()
    
    skf = StratifiedKFold(n_splits=nfold, shuffle=True)
    y_test_total = []
    y_pred_total = []

    for train_index, test_index in skf.split(document_term_matrix, labels):
        X_train, X_test = document_term_matrix[train_index], document_term_matrix[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_test_total.extend(y_test.tolist())
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_total.extend(y_pred.tolist())
        p,r,f,s = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print accuracy_score(y_test, y_pred)
        a_score.append(accuracy_score(y_test, y_pred))
        precision.append(p)
        recall.append(r)
        fscore.append(f)
    
    plot_learning_curve(clf, "Learning Curves", document_term_matrix, labels, ylim=None, cv=skf, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))

    plt.savefig('lc.png')

    return pd.Series(y_test_total), pd.Series(y_pred_total), np.mean(precision),np.mean(recall),np.mean(fscore), np.mean(a_score)

titles, labels = loadData()
processed_titles, processed_titles_wordlist = preProcessing(titles)
dtm,vect = getDTMByTFIDF(processed_titles,None)

chisqDtm, chisqModel = featuresByChiSq(dtm,labels,2000)
#igDtm = featuresByInformationGain(dtm,labels)
#lsaDtm = featuresByLSA(dtm,100)


num_features = 300    # Word vector dimensionality                      
min_word_count = 1    # Minimum word count                        
num_workers = 1       # Number of threads to run in parallel
context = 8           # Context window size                                                                                    
downsampling = 1e-5   # Downsample setting for frequent words

word2vec_model = Word2Vec(processed_titles_wordlist, workers=num_workers, 
            size=num_features, min_count = min_word_count, 
            window = context, sample = downsampling)
word2vec_model.init_sims(replace=True)

wordVecs = getAvgFeatureVecs(processed_titles_wordlist, word2vec_model, num_features)


#Combine features from chiSq and word2Vec
combinedFeatures = np.hstack([chisqDtm,wordVecs])



iterations = [i for i in range(1, 16)]
y_test, y_pred, precision, recall, fscore, accuracy = crossValidate(chisqDtm,labels,"NN",15)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
    
class_names = ['INFOCOM', 'ISCAS', 'SIGGRAPH', 'VLDB', 'WWW']
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion Matrix')
print a_score 
print "\nHere is the Classification Report:"
print classification_report(y_test, y_pred, target_names=class_names)

print "ChiSq Features:",precision, recall, fscore, accuracy

plt.tight_layout()
plt.savefig("imga.png")
plt.figure()
plt.plot(iterations, a_score)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

#a_score = []
#iterations = [i for i in range(0, 15)]
#y_test, y_pred, precision, recall, fscore, accuracy = crossValidate(chisqDtm,labels,"LR",15)

#plt.plot(iterations, a_score, 'r')
#plt.show()
#precision, recall, fscore = crossValidate(combinedFeatures,labels,"SVM",10)
#print "ChiSq Features:",precision, recall, fscore

plt.savefig("img.png")

# iterations vs accuracy

# Frequency of papers by conferences

# Real life data. 

# What kinda parameters should we visualize?
