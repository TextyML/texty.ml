import nltk # language processing
from sklearn.base import BaseEstimator
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import StanfordNERTagger
import operator

class NamedEntityVectorizer(BaseEstimator):
    
    def __init__(self, entity="PERSON", max_features=None): #max_df=1.0, min_df=1
        #self.max_df = max_df
        #self.min_df = min_df
        self.entity = entity
        self.max_features = max_features
        self.features = []
        self._v = {}
        self._c = {}
        
    def get_feature_names(self):
        return np.array(self.features)

    def _stanford_tagger(self, text):
        ttext = word_tokenize(text)
        st_tags = StanfordNERTagger('/opt/nltk_data/stanford/classifiers/english.all.3class.distsim.crf.ser.gz',
                                   '/opt/nltk_data/stanford/stanford-ner.jar', encoding='utf-8')
        tagged = st_tags.tag(ttext)  

        tagList = []
        tempTag = None
        for tag_index, tag in enumerate(tagged):   
            if tag[1] != "O":
                if tempTag is None:
                    tempTag = tag
                elif tempTag[1] == tag[1]:
                    tempTag = (tempTag[0] + " "+ tag[0], tempTag[1])
                else:
                    tagList.append(tempTag)
                    tempTag = tag
            elif tempTag is not None:
                tagList.append(tempTag)
                tempTag = None

        return(tagList)

    def _count_vocab(self, raw_documents, build_doc):
        vocabulary = {}
        doc_counter = []
        counter = {}
        for doc in raw_documents:
            tagged = self._stanford_tagger(doc)
            list_of = [chunk for chunk in tagged if chunk[1] == self.entity]
            list_set_of = list(set(list_of))
            if build_doc:
                doc_dict = {el:0.0 for el in self.features}
                        
            for feature in list_of:
                try:
                    if build_doc:
                        doc_dict[feature[0]] += 1
                    vocabulary[feature[0]] += 1
                except KeyError:
                    vocabulary[feature[0]] = 1
                    
            for feature in list_set_of:
                try:
                    counter[feature[0]] += 1
                except KeyError:
                    counter[feature[0]] = 1
                    
            if build_doc:        
                doc_counter.append(doc_dict)
                        
        return vocabulary, counter, doc_counter
        
    def fit(self, raw_documents, y=None):
        v, c, d = self._count_vocab(raw_documents, False)
        if self.max_features is None or len(v) > self.max_features:
            max = len(v)
        sorted_v = sorted(v.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.features = [p[0] for p in sorted_v]
        self._v = v
        self._c = c
        return self
    
    def transform(self, raw_documents):
        v, c, d = self._count_vocab(raw_documents, True)
        result = np.array([list(doc.values()) for doc in d])
        lengthes = [len(word_tokenize(document)) for document in raw_documents]
        result_norm = [res / lengthes[ind] for ind, res in enumerate(result)]
        return result_norm