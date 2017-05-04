import nltk # language processing
from sklearn.base import BaseEstimator
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import operator
import socket
from .DebugLogger import debug_log
import collections
from nltk import word_tokenize, pos_tag, ne_chunk

class NamedEntityVectorizer(BaseEstimator):
    
    def __init__(self, entity="PERSON", host="172.17.0.1", port=9000, max_features=None, stanford_tagger=True): #max_df=1.0, min_df=1
        #self.max_df = max_df
        #self.min_df = min_df
        self.stanford_tagger = stanford_tagger
        self.entity = entity
        self.connection_addr = (host, port)
        self.max_features = max_features
        self.features = []
        self._v = {}
        self._c = {}
        
    def get_feature_names(self):
        return np.array(self.features)

    def _tagger(self, text):
        
        if self.stanford_tagger:
            from pycorenlp import StanfordCoreNLP
            nlp = StanfordCoreNLP('http://172.17.0.1:9000')
            text = (text)
            output = nlp.annotate(text, properties={
                  'annotators': 'ner',
                  'outputFormat': 'json'
            })

            result = []
            for word in output['sentences'][0]['tokens']:
                result.append([word['ner'],word['word']])
        else:
            result = [[tag[1],tag[0][0]] for tag in ne_chunk(pos_tag(word_tokenize(text))).pos()]
        
        tagList = []
        tempTag = None
        for tag_index, tag in enumerate(result):
            if len(tag) < 2 :
                continue
            if tag[0] not in ["O","S"]:
                if tempTag is None:
                    tempTag = tag
                elif tempTag[0] == tag[0]:
                    tempTag = [tempTag[0], tempTag[1] + " "+ tag[1]]
                else:
                    tagList.append(tempTag)
                    tempTag = tag
            elif tempTag is not None:
                tagList.append(tempTag)
                tempTag = None

        return(tagList)
    
    def _nltk_tagger(self, doc):
        tagged = ne_chunk(pos_tag(word_tokenize(doc)))
        tagList = []
        tempTag = None
        for tag_index, nltk_tag in enumerate(tagged.pos()):
            tag = [nltk_tag[1], nltk_tag[0][0]]
            if tag[0] in ["PERSON","ORGANIZATION"]:
                if tempTag is None:
                    tempTag = tag
                elif tempTag[0] == tag[0]:
                    tempTag = [tempTag[0], tempTag[1] + " "+ tag[1]]
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
            tagged = self._tagger(doc)
            list_of = [chunk[1] for chunk in tagged if chunk[0] == self.entity]
            list_set_of = list(set(list_of))
            if build_doc:
                doc_dict = {el:0.0 for el in self.features}
                        
            for feature in list_of:
                try:
                    if build_doc:
                        doc_dict[feature] += 1
                    vocabulary[feature] += 1
                except KeyError:
                    vocabulary[feature] = 1
                    
            for feature in list_set_of:
                try:
                    counter[feature] += 1
                except KeyError:
                    counter[feature] = 1
                    
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
        sorted_d = []
        for i, doc_entry in enumerate(d):  
            sorted_d.append(collections.OrderedDict())
            for feature in self.features:
                sorted_d[i][feature] = doc_entry[feature] 
        result = np.array([list(doc.values()) for doc in sorted_d])
        lengthes = [len(word_tokenize(document)) for document in raw_documents]
        result_norm = [res / lengthes[ind] for ind, res in enumerate(result)]
        return result_norm