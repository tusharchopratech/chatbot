import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


def bow(tokenized_sentence, all_words):
    """
    sentence = ["word_1","word_2",word_3"]
    all_words = ["word_1","word_2",word_3","word_4","word_5",word_6"]
    bog = [1,1,1,0,0,0]
    """
    stemmed_words = [stem(w) for w in tokenied_sentence]
    bow = np.zeros(len(all_words))
    for i in range(len(all_words)):
        if all_words[i] in stemmed_words:
            bow[i]=1
    return bow

# test = "How long does it take for delivery?"
# print(test)
# tokenized_sentence = tokenize(test)
# print(tokenized_sentence)
# s_list = [stem(w) for w in tokenized_sentence]
# print(s_list)
# print()