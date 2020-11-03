import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


def bow(tokenized_sentence, all_words):
    pass


test = "How long does it take for delivery?"
print(test)
tokenized_sentence = tokenize(test)
print(tokenized_sentence)
s_list = [stem(w) for w in tokenized_sentence]
print(s_list)
# print()