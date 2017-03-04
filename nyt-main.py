from Import.Preprocessor import Preprocessor
from Import.LinguisticVectorizer import LinguisticVectorizer
from Import.NamedEntityVectorizer import NamedEntityVectorizer
from Import.CorpusImporter import CorpusImporter
from sklearn.feature_selection import SelectKBest
import pickle 

corpus = CorpusImporter()
corpus.clearMemory()
corpus.crawlNYT(per_tag=10, is_multilabel=False)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split([news.text for news in corpus._Collection], [news.tags[0] for news in corpus._Collection], test_size=0.25, random_state=42)

class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        print(X.shape)
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            self.k = "all"

grid = pickle.load( open( "grid_pre.pickle", "rb" ) )
print("Starting fitting ...")
grid.fit(X_train, Y_train)


