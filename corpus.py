import os
import sys

import dill
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from Import.CorpusImporter import CorpusImporter

corpus = CorpusImporter()
corpus.clearMemory()
corpus.crawlNYT(per_tag=1001, is_multilabel=False, nytPaths = ["2007","2006","2005","2004","2003","2002","2001","2000","1999","1998","1997","1996","1995","1994"])

with open('corpus.pickle', 'wb') as handle:
    corpus = dill.dump(corpus._Collection, handle, protocol=dill.HIGHEST_PROTOCOL)
