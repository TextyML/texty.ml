import nltk
import math
from sklearn.base import BaseEstimator
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import string

import re
from collections import Counter
def words(text): return re.findall(r'\w+', text.lower())
dictionary = Counter(words(open('/home/retkowski/Data/dicts/dict_all.txt').read()))


# mehr Wortarten…

class LinguisticVectorizer(BaseEstimator):

    def get_feature_names(self):
        return np.array(
            ['text_length',
             'number_of_paragraphs',
             'average_sent_length',
             'average_word_length',
             'number_of_nouns',
             'number_of_adjectives',
             'number_of_verbs',
             'number_of_ex_there',
             'number_of_foreign_words',
             'number_of_modal_words',
             'number_of_judging_adjectives',
             'number_of_numbers',
             'type_token_relation',
             'concentration_index',
             'hapaxes_index',
             'action_index',
             #'number_of_spelling_mistakes',
             'number_of_question_marks',
             'number_of_exclamations',
             'number_of_percentages',
             'number_of_currency_symbols',
             'number_of_paragraph_symbols',
             'content_fraction',
             'number_of_cappsed_words',
             'number_of_first_person_pronouns']
        )

    def fit(self, documents, y=None):
        return self
    
    def __filter(self, string):
        return [self.__remove_punctuation(w) for w in word_tokenize(string) if self.__remove_punctuation(w).isalpha()]
    
    def __remove_punctuation(self, s):
        return s.translate(str.maketrans("", "", string.punctuation))
    
    def _get_text_length(self, string):
        tokens = self.__filter(string)
        return len(tokens)
    
    def _get_number_of_paragraphs(self, string):
        return round(string.count('\n') / 2)
    
    def _get_average_sent_length(self, string):
        tokens = self.__filter(string)
        if len(sent_tokenize(string)) == 0:
            return len(tokens)
        return len(tokens) / len(sent_tokenize(string))
    
    def _get_average_word_length(self, string):
        tokens = self.__filter(string)
        word_length_list = []
        for word in tokens:
            word_length_list.append(len(word))
        return np.average(word_length_list)
    
    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    
    def _get_number_of_numbers(self, string):
        return len([w for w in word_tokenize(string) if self.is_number(self.__remove_punctuation(w))])
    
    def _get_number_of_tagged_words(self, string, tags):
        tagged_words = [a[0] for a in pos_tag(self.__filter(string)) if a[1] in tags]
        return len(tagged_words) / self._get_text_length(string)
    
    def _get_number_of_adjectives(self, string):
        return self._get_number_of_tagged_words(string,['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'])
       
    def _get_number_of_verbs(self, string):
        return self._get_number_of_tagged_words(string,['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
    
    # Stopwords?
    def _get_ttr(self, string):
        tokens = self.__filter(string)
        if len(tokens) == 0:
            return 0
        return len(set(tokens)) / len(tokens)

    def _get_aq(self, string):
        adjectives = self._get_number_of_adjectives(string)
        verbs = self._get_number_of_verbs(string)
        if adjectives == 0:
            return verbs
        return verbs / adjectives

    def _get_naq(self, string):
        adjectives = self._get_number_of_adjectives(string)
        verbs = self._get_number_of_verbs(string)
        if adjectives == 0 and verbs == 0:
            return 0.0
        return verbs / (adjectives + verbs)

    def _get_hl(self, string):
        words = self.__filter(string)
        fdist = nltk.FreqDist(words)
        hapaxes = fdist.hapaxes()
        if len(words) == 0:
            return len(hapaxes)
        return len(hapaxes) / len(words)

    def _get_koi(self, string, n):
        words = self.__filter(string)
        fdist = nltk.FreqDist(words)
        sum = 0
        for word in fdist.most_common(n):
            sum += word[1]
        if len(words) == 0:
            return sum
        return sum / len(words)

    def _get_nkoi(self, string, n, m):
        words = self.__filter(string)
        h = math.floor(len(words) / m)
        if h == 0:
            return self._get_koi(string, 15)
        sum = 0
        for i in range(h):
            sum += self._get_nkoi_i(words[i*m:(i+1)*m],n)
        return (sum/h)

    def _get_nkoi_i(self, words, n):
        fdist = nltk.FreqDist(words)
        sum = 0
        for word in fdist.most_common(n):
            sum += word[1]
        if len(words) == 0:
            return sum
        return sum / len(words)
    
    def _get_number_of_spelling_mistakes(self, string):
        text_vocub = set(w.lower() for w in word_tokenize(string) if w.isalpha())
        text_dict  = set(w.lower() for w in dictionary)
        return len(text_vocub - text_dict) / self._get_text_length(string)
    
    def _get_number_of_currency_symbols(self, string):
        currencies = ["£","€","$","¥","¢","₩"]
        sum = 0
        for currency in currencies:
            sum += self._get_number_of_symbol(string, currency)
        return sum / self._get_text_length(string)
    
    def _get_number_of_symbol(self, string, symbol):
        return string.count(symbol) / self._get_text_length(string)
    
    def _get_content_fraction(self, string):
        tokens = self.__filter(string)
        content = [w for w in tokens if w.lower() not in stopwords.words('english')]
        if len(tokens) == 0:
            return 0
        return len(content) / len(tokens)
    
    def _get_number_of_cappsed_words(self, string):
        tokens = self.__filter(string)
        return np.sum([t.isupper() for t in tokens if len(t) > 2]) / self._get_text_length(string)
    
    def _get_number_of_first_person_pronouns(self, string):
        tokens = word_tokenize(string)
        pronouns = ["i","me","my", "mine", "myself","we", "our", "ours", "ourself"]
        sum = 0
        mode = 0
        for word in tokens:
            if word == "``":
                mode = mode + 1
            elif word == "''":
                mode = mode - 1
            
            if mode <= 0 and word.lower() in '\t'.join(pronouns):
                sum += 1
        return sum / len(tokens)

    def transform(self, documents):
        text_length = [self._get_text_length(d) for d in documents]
        number_of_paragraphs = [self._get_number_of_paragraphs(d) for d in documents]
        average_length_of_sent = [self._get_average_sent_length(d) for d in documents]
        average_word_length = [self._get_average_word_length(d) for d in documents]
        number_of_nouns = [self._get_number_of_tagged_words(d,['NN', 'NNS', 'NNP', 'NNPS']) for d in documents]
        number_of_adjectives = [self._get_number_of_tagged_words(d,['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']) for d in documents]
        number_of_verbs = [self._get_number_of_tagged_words(d,['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) for d in documents]
        number_of_foreign_words = [self._get_number_of_tagged_words(d,['FW']) for d in documents]
        number_of_modal_words = [self._get_number_of_tagged_words(d,['MD']) for d in documents]
        number_of_judging_adjectives = [self._get_number_of_tagged_words(d,['JJR', 'JJS', 'RBR', 'RBS']) for d in documents]
        number_of_ex_there = [self._get_number_of_tagged_words(d,['EX']) for d in documents]
        type_token_relation = [self._get_ttr(d) for d in documents]
        concentration_index = [self._get_nkoi(d,10,150) for d in documents]
        hapaxes_index = [self._get_hl(d) for d in documents]
        action_index = [self._get_naq(d) for d in documents]
        #number_of_spelling_mistakes = [self._get_number_of_spelling_mistakes(d) for d in documents]
        number_of_question_marks = [self._get_number_of_symbol(d, "?") for d in documents]
        number_of_exclamations = [self._get_number_of_symbol(d, "!") for d in documents]
        number_of_percentages = [self._get_number_of_symbol(d, "%") for d in documents]
        number_of_currency_symbols = [self._get_number_of_currency_symbols(d) for d in documents]
        number_of_paragraph_symbols = [self._get_number_of_symbol(d, "§") for d in documents]
        content_fraction = [self._get_content_fraction(d) for d in documents]
        number_of_cappsed_words = [self._get_number_of_cappsed_words(d) for d in documents]
        number_of_first_person_pronouns = [self._get_number_of_first_person_pronouns(d) for d in documents]
        number_of_numbers = [self._get_number_of_numbers(d) for d in documents]

        result = np.array(
            [text_length,
             number_of_paragraphs,
             average_length_of_sent,
             average_word_length,
             number_of_nouns,
             number_of_adjectives,
             number_of_verbs,
             number_of_ex_there,
             number_of_foreign_words,
             number_of_modal_words,
             number_of_judging_adjectives,
             number_of_numbers,
             type_token_relation,
             concentration_index,
             hapaxes_index,
             action_index,
             #number_of_spelling_mistakes,
             number_of_question_marks,
             number_of_exclamations,
             number_of_percentages,
             number_of_currency_symbols,
             number_of_paragraph_symbols,
             content_fraction,
             number_of_cappsed_words,
             number_of_first_person_pronouns]
        ).T

        return result