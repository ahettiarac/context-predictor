import gensim
from preprocessor import PreProccessor
from gensim import corpora
import pickle

class TopicModel:

    topicModel = None
    tokens = []
    preprocessor = None
    num_topics = 5

    def __init__(self,text,topics=5):
        self.preprocessor = PreProccessor(text)
        self.num_topics = topics

    def prepareTextForLda(self):
        self.tokens = self.preprocessor.tokenize()
        self.tokens = [token for token in self.tokens if len(token) > 4]
        self.tokens = [token for token in self.tokens if not self.preprocessor.filterStopWords(token)]
        self.tokens = [self.preprocessor.get_lemma(token) for token in self.tokens]

    def trainModel(self):
        corpus,dictionary = self.prepareCorpora(self.tokens)
        pickle.dump(corpus,open('corpus.pkl','wb'))
        dictionary.save('dictionary.gensim')
        ldaModel = gensim.models.ldamodel.LdaModel(corpus,num_topics=self.num_topics,
                    id2word=dictionary,passes=15)
        model_name = 'model5.gensim'            
        ldaModel.save(model_name)
        topics = ldaModel.print_topics(num_words=4)
        for topic in topics:
            print(topic)

    def prepareCorpora(self,words):
        dictionary = corpora.Dictionary(words)
        corpus = [dictionary.doc2bow(text) for text in words]
        return corpus,dictionary            

    def predictTopic(self,model_name,preprocessed_tokens):
        ldaModel = gensim.models.ldamodel.LdaModel.load(model_name)
        corpus =  self.prepareCorpora(preprocessed_tokens)
        topics = ldaModel.get_document_topics(corpus)
        print(topics)
        return topics




