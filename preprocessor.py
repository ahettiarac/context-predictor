import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

class PreProccessor:

    paragraph = ''
    parser = None
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