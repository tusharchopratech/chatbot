"""
NLP utilities like stemming, tokanization and bag of words
"""

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bow(tokenized_sentence, all_words):
    """
    sentence = ["word_1","word_2",word_3"]
    all_words = ["word_1","word_2",word_3","word_4","word_5",word_6"]
    bow = [1,1,1,0,0,0]
    """
    stemmed_words = [stem(w) for w in tokenized_sentence]
    bow = np.zeros(len(all_words), dtype=np.float32)
    for i in range(len(all_words)):
        if all_words[i] in stemmed_words:
            bow[i]=1
    return bow
