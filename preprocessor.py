import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import operator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

class PreProccessor:

    paragraph = ''
    parser = None
    unique_labels = []
    def __init__(self,text):
        self.paragraph = text
        self.parser = English()

    def tokenize(self):
        lda_tokens = []
        tokens = self.parser(self.paragraph)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif len(token) < 3:
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens

    def word_frequency(self,corpus=[[]]):
        frequency = defaultdict(int)
        for doc in corpus:
            for w in doc:
                frequency[w] += 1
        return dict(sorted(frequency.items(),key=operator.itemgetter(1),reverse=True))        

    def get_lemma(self,word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma 

    def get_lemma2(self,word):
        return WordNetLemmatizer().lemmatize(word)   

    def filterStopWords(self,word):
        stopWords = set(nltk.corpus.stopwords.words('english'))
        if word in stopWords:
            return True
        else:
            return False

    def filter_uniques(self,word):
        if self.unique_labels.index(word) == -1:
            self.unique_labels.append(word)
        return word

    def prepare_text_for_train_rnn(self,text_corpus):
        self.unique_labels = []
        labels = text_corpus['label'].apply(lambda x:self.filter_uniques(x))
        unique_integer_labels = []
        index = 0
        for label in self.unique_labels:
            unique_integer_labels[index]
            index+=1
        labels = np.asarray(unique_integer_labels, dtype=int) 
        X_train,X_test,Y_train,Y_test = train_test_split(text_corpus['text'],labels,test_size=0.2)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train) 
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        X_train = pad_sequences(X_train, 50)
        X_test = pad_sequences(X_test, 50)
        return X_train, X_test, Y_train, Y_test

    def prepare_text_for_predict(self,text_corpus):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_corpus)
        X_Test = tokenizer.texts_to_sequences(text_corpus)
        X_Test = pad_sequences(X_Test, 50)
        return X_Test      
