from Import.Preprocessor import Preprocessor
from Import.LinguisticVectorizer import LinguisticVectorizer
from Import.NamedEntityVectorizer import NamedEntityVectorizer
from Import.CrawlerImporter import CrawlerImporter

importer = CrawlerImporter()
importer.importAllFromDB()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split([news.text for news in importer._Collection], [news.tags[0] for news in importer._Collection], test_size=0.25, random_state=42)
print(len(X_train), " Newspaper Articles in the Training Set")
print(len(X_test), " Newspaper Articles in the Test Set")

from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer 
from nltk import PorterStemmer
from nltk import LancasterStemmer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import re
from collections import Counter
def words(text): return re.findall(r'\w+', text.lower())
dictionary = Counter(words(open('/home/retkowski/Data/dicts/dict_all.txt').read()))

#preprocessor = Preprocessor.Preprocessor(stopwords = stopwords.words('english'), stemmer = SnowballStemmer("english"))
preprocessor = Preprocessor(stopwords = stopwords.words('english'), stemmer = SnowballStemmer("english"))

# max_df = 0.75 for corpus-specific stopwords
cv = CountVectorizer(max_df=0.75, ngram_range=(1,1), max_features=1000, preprocessor=preprocessor.get_preprocessed_text)
tt = TfidfTransformer()
lv = LinguisticVectorizer()
#pv = NamedEntityVectorizer.NamedEntityVectorizer(max_features=100, entity="PERSON")
clf = MultinomialNB() # GaussianNB
clf = LogisticRegression()
#svd = TruncatedSVD(n_components=900, n_iter=7, random_state=42)
#select = SelectKBest(score_func=f_classif, k=750)
#fit = test.fit(fu.fit_transform(X_train), Y_train)
class NewSVD(TruncatedSVD):
    def fit_transform(self, X, y=None):
        if self.n_components < X.shape[1]:
            return super().fit_transform(X, y)
        else:
            random_state = check_random_state(self.random_state)
            U, Sigma, VT = randomized_svd(X, X.shape[1],
                                          n_iter=self.n_iter,
                                          random_state=random_state)
            self.components_ = VT
            #X_transformed = U * Sigma
            return X.toarray()
        
svd = NewSVD(n_components=900, n_iter=7, random_state=42)

class SelectAtMostKBest(SelectKBest):

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"
            
select = SelectAtMostKBest(score_func=f_classif, k=750)

fu = FeatureUnion([
    ('ngram_tf_idf', Pipeline([
      ('counts', cv),
      ('tf_idf', tt),
      ('scaler', StandardScaler(with_mean=False)),
      ('mascaler', MaxAbsScaler())
    ])),
    ('lv', Pipeline([
      ('linguistic', lv),
      ('scaler', StandardScaler(with_mean=False)),
      ('mascaler', MaxAbsScaler())
    ]))#,
    #('pv', Pipeline([
    #  ('person', pv),
    #  ('scaler', StandardScaler(with_mean=False)),
    #  ('mascaler', MaxAbsScaler())
    #]))
])

pipeline = Pipeline([
  ('features', fu),
  ('dim_red', svd),
  #('scaler', StandardScaler(with_mean=False)),
  #('mascaler', MaxAbsScaler()),
  ('k_best', select),
  ('classifier', clf)
])

param_grid = [
    {
        #'k_best__k': [200,500,800,1000],
        'k_best__k' : [800],
        #'dim_red__n_components': [100, 500, 900, 1200],
        'dim_red__n_components': [100, 600],
        #'features__ngram_tf_idf__counts__max_features': [500,800,1000,1200,1400],
        'features__ngram_tf_idf__counts__max_features': [500,800],
        #'features__ngram_tf_idf__counts__ngram_range': [(1,1),(1,2),(1,3)],
        'features__ngram_tf_idf__counts__ngram_range': [(1,1)]
        #'features__pv__person__max_features': [50,75,100,125,150]
    }
]

print(pipeline.get_params().keys())
grid = GridSearchCV(pipeline, cv=3, n_jobs=-1, param_grid=param_grid, verbose=100)

print("Starting fitting ...")
grid.fit(X_train, Y_train)

with open('grid.pickle', 'wb') as fp:
    pickle.dump(grid, fp)
