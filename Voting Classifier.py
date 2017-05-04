
# coding: utf-8

# # Import

# In[1]:

import os
import sys
import dill
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

ownModules = ['Import.CorpusImporter',
              'Import.Preprocessor',
              'Import.LinguisticVectorizer',
              'Import.NamedEntityVectorizer']

for module in sys.modules:
    if module in ownModules:
        del sys.modules[module]
    
#from Import.CorpusImporter import CorpusImporter
from Import.Preprocessor import Preprocessor
from Import.LinguisticVectorizer import LinguisticVectorizer
from Import.NamedEntityVectorizer import NamedEntityVectorizer


# # Korpus Import

# In[ ]:

Collection = []
with open('corpus.pickle', 'rb') as handle:
    Collection = dill.load(handle)
print(str(len(Collection)) + " Artikel eingelesen.")
print(Collection[0].titles)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split([news.text for news in Collection if news.text is not None], [news.tags[0] for news in Collection if news.text is not None], test_size=0.11, random_state=42)
print(len(X_train), " Newspaper Articles in the Training Set")
print(len(X_test), " Newspaper Articles in the Test Set")


# # All Imports

# In[6]:

# general
import gensim

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif

# nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer 
from nltk import PorterStemmer, LancasterStemmer


# Numpy
import numpy as np
import re
from collections import Counter

preprocessor = Preprocessor(stopwords = stopwords.words('english'), stemmer = SnowballStemmer("english"))


# # Word2Vec

# In[9]:

word2vec = gensim.models.KeyedVectors.load_word2vec_format(os.getcwd() + "/../Word2Vec/glove_model2.txt", binary=False)
word2vec.init_sims(replace=True)


# In[7]:

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 300
        self.stopwordsList = stopwords.words('english')
    
    def fit(self, X, y):
        return self
    
    def _remove_stopwords(self, document):
        return [word for word in document if word not in self.stopwordsList]
    
    def _mean(self, X):     
        return np.mean([self.word2vec[w] for w in self._remove_stopwords(X.lower().strip().split(' ')) if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
    
    
    def transform(self, documents):         
        return np.array(
            [self._mean(d) for d in documents]
        )


# In[8]:

w2v_fu = FeatureUnion([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(word2vec))
])

w2v_svc_ppl = Pipeline([
  ('features', w2v_fu),
  ('scaler', StandardScaler(with_mean=True)),
  ('classifier', SVC(kernel='rbf', probability=True, verbose=100, C = 1000))
])


# # General Feature Union

# In[ ]:

fu = FeatureUnion([
    ('ngram_tf_idf', Pipeline([
      ('counts', CountVectorizer(max_df=0.75, ngram_range=(1,3), max_features=2000, preprocessor=preprocessor.get_preprocessed_text)),
      ('tf_idf', TfidfTransformer())
    ])),
    ('lv', Pipeline([
      ('linguistic', LinguisticVectorizer())
    ]))
])


# # SVM

# In[ ]:

svm_fu = FeatureUnion([
    ('ngram_tf_idf', Pipeline([
      ('counts', CountVectorizer(max_df=0.75, ngram_range=(1,3), max_features=2000, preprocessor=preprocessor.get_preprocessed_text)),
      ('tf_idf', TfidfTransformer())
    ])),
    ('lv', Pipeline([
      ('linguistic', LinguisticVectorizer())
    ])),
    ('w2v', Pipeline([
        ("mean_embedding", MeanEmbeddingVectorizer(word2vec))
    ]))
])

svc_high_ppl = Pipeline([
  ('features', svm_fu),
  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
  ('scaler', StandardScaler(with_mean=True)),
  ('k_best', SelectKBest(score_func=f_classif, k=1800)),
  ('dim_red', PCA(n_components=1500)),
  ('classifier', SVC(kernel='rbf', probability=True, verbose=100, C=1000))
])


# # MLP

# In[ ]:

mlp_fu = FeatureUnion([
    ('ngram_tf_idf', Pipeline([
      ('counts', CountVectorizer(max_df=0.75, ngram_range=(1,3), max_features=4000, preprocessor=preprocessor.get_preprocessed_text)),
      ('tf_idf', TfidfTransformer())
    ])),
    ('lv', Pipeline([
      ('linguistic', LinguisticVectorizer())
    ]))
])

mlp_clf = MLPClassifier(hidden_layer_sizes=(300,300),solver='adam',activation='relu',learning_rate_init=0.01,max_iter=750,verbose=True)

mlp1_pipeline = Pipeline([
  ('features', mlp_fu),
  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
  ('scaler', StandardScaler(with_mean=True)),
  ('classifier', mlp_clf)
])


# # Naiver Bayes

# In[ ]:

mnb_ppl = Pipeline([
  ('features', fu),
  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
  ('scaler', MinMaxScaler()),
  ('k_best', SelectKBest(score_func=f_classif, k=1500)),
  ('dim_red', PCA(n_components=1200)),
  ('scaler2', MinMaxScaler()),
  ('classifier', MultinomialNB())
])


# # Random Forest

# In[ ]:

random_200_ppl = Pipeline([
  ('features', fu),
  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
  ('scaler', StandardScaler(with_mean=True)),
  ('k_best', SelectKBest(score_func=f_classif, k=1500)),
  ('dim_red', PCA(n_components=1200)),
  ('classifier', RandomForestClassifier(verbose=100, n_estimators=200))
])


# # Logistic Regression

# In[ ]:

logistic_pipeline = Pipeline([
  ('features', fu),
  ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
  ('scaler', StandardScaler(with_mean=True)),
  ('k_best', SelectKBest(score_func=f_classif, k=1500)),
  ('dim_red', PCA(n_components=1200)),
  ('classifier', LogisticRegression(verbose=100))
])


# # Voting Classifier

# In[ ]:

eclf = VotingClassifier(estimators=[('word2vec', w2v_svc_ppl), ('svm',svc_high_ppl),('mlp',mlp1_pipeline),('nb', mnb_ppl), ('rf', random_200_ppl), ('lr', logistic_pipeline)], voting='soft', weights=[4,2,2,1,1,1])


# # Fitting 

# In[ ]:

eclf.fit(X_train,Y_train)


# # Evaluation

# In[ ]:

eclf_predicted = eclf.predict(X_test) 
print(metrics.accuracy_score(Y_test, eclf_predicted))
print(metrics.classification_report(Y_test, eclf_predicted))
print(metrics.cohen_kappa_score(Y_test, eclf_predicted))
metrics.confusion_matrix(Y_test, eclf_predicted)


# # Export with Dill

# In[ ]:

with open('eclf.pickle', 'wb') as handle:
    dill.dump(eclf, handle, protocol=dill.HIGHEST_PROTOCOL)

