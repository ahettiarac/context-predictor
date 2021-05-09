import gensim
from preprocessor import PreProccessor
from gensim import corpora
import pickle
from keras import Sequential
from keras.layers import Dense,LSTM,Embedding,Flatten

class TopicModel:

    topicModel = None
    tokens = []
    preprocessor = None
    num_topics = 5
    model = None

    def __init__(self,text,topics=5):
        self.preprocessor = PreProccessor(text)
        self.num_topics = topics

    def prepareTextForLda(self):
        self.tokens = self.preprocessor.tokenize()
        freq_tokens = self.preprocessor.word_frequency(self.tokens)
        self.tokens = [[token for token in sentence if freq_tokens[token] > 5] for sentence in self.tokens]
        self.tokens = [token for token in self.tokens if len(token) > 4]
        self.tokens = [token for token in self.tokens if not self.preprocessor.filterStopWords(token)]
        self.tokens = [self.preprocessor.get_lemma2(token) for token in self.tokens]

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

    def initializeRNN(self,num_unique_words, max_length):
        self.model = Sequential()
        self.model.add(Embedding(num_unique_words, 32, input_length=max_length))
        self.model.add(LSTM(32, dropout=0, recurrent_dropout=0))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()

    def trainRNN(self, X_Train, X_Test, Y_Train, Y_Test):
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.fit(X_Train, Y_Train, batch_size=32, epochs=10, validation_data=(X_Test, Y_Test))
        self.model.save_weights('trained_model.h5')

    def predictVector(self,corpus,model_name):
        self.model.load_weights(model_name)
        vector_prediction = self.model.predict(corpus)
        return vector_prediction

