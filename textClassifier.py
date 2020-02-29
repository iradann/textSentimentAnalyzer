import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import *

import gensim
from collections import defaultdict
from stop_words import safe_get_stop_words

from sklearn.model_selection import train_test_split

from termcolor import colored


stop_words = safe_get_stop_words('russian')

data = pd.read_excel('reviews.xlsx')
data.columns = ['rate', 'text']

data.text = data.text.str.lower()

data.rate.replace(to_replace = [1, 2], value = -1, inplace = True)
data.rate.replace(to_replace = [3], value = 0, inplace = True)
data.rate.replace(to_replace = [4, 5], value = 1, inplace = True)

y = data.rate
x = data.text.values

cvec = CountVectorizer(stop_words=stop_words)
x_cvec = cvec.fit_transform(x)


tvec = TfidfVectorizer(stop_words=stop_words)
x_tvec = tvec.fit_transform(x)

model = gensim.models.Word2Vec(x, size=500)
w2v = dict(zip(model.wv.index2word, model.wv.vectors))

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    
    def fit(self, X, y):
        tfidf = TfidfVectorizer(stop_words=stop_words)
        tfidf.fit(X)
        
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


w2v2 = TfidfEmbeddingVectorizer(w2v)
w2v2.fit(x, y)
x_x2v = w2v2.transform(x)


nb = MultinomialNB()
kn = KNeighborsClassifier()
lr = LogisticRegression(solver = 'lbfgs', multi_class= 'multinomial')
svm = SVC(gamma = 'scale')
rf = RandomForestClassifier()


for data in zip(['CountVectorizer', 'TfidfVectorizer', 'Word2Vec'],[x_cvec, x_tvec, x_x2v]):
    print(colored('Vectorization: {}\n'.format(data[0]), 'green'))
    X_train, X_test, y_train, y_test = train_test_split(data[1], y, test_size=0.1, random_state=42)

    for method in zip(['Multinomial Nauve Bayes', 'K-nearest', 'Logistic Regression', 'SVM', 'Random forest'], [nb, kn, lr, svm, rf]):
   
        if (method[0]== 'Multinomial Nauve Bayes') & (data[0]== 'Word2Vec'):
            method[1].fit(X = X_train + 10, y = y_train)
            predict = method[1].predict(X_test + 10)
        else: 
            method[1].fit(X = X_train, y = y_train)
            predict = method[1].predict(X_test)
        
        balanced_acc = balanced_accuracy_score(y_test, predict)
        mat_corr = matthews_corrcoef(y_test, predict)
        cohen_score= cohen_kappa_score(y_test, predict)
        confu_matrix= confusion_matrix(y_test, predict)
    
        print('Method: {} \nbalanced_accuracy_score = {:.1%} \nmatthews_corrcoef = {:.1%} \ncohen_kappa_score = {:.1%} \nconfusion_matrix =  \n{} \n'.format(
                method[0], balanced_acc, mat_corr, cohen_score, confu_matrix))
        
