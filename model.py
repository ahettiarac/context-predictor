import nltk
from random import shuffle
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
import pickle

sia = SentimentIntensityAnalyzer()
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

stopwords = nltk.corpus.stopwords.words("english")
stopwords.extend([w.lower() for w in nltk.corpus.names.words()])
#words = [w for w in words if w.lower() not in stopwords]

#test = ""
#words: list[str] = nltk.word_tokenize(text)

#frequencyDistribution = nltk.FreqDist(words)
#frequencyDistribution.most_common(3)
#frequencyDistribution.tabulate(3)

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in stopwords:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]

negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]

positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}

def extract_features(text):
    features = dict()
    word_count = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                word_count += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])        

    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = word_count

    return features

#features = [(extract_features(nltk.corpus.movie_reviews.raw(review)),"pos")
 #           for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])]

#features.extend([(extract_features(nltk.corpus.movie_reviews.raw(review)),"neg")
 #               for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])])

def train_classifier(features):
    train_count = len(features)
    shuffle(features)
    classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
    f = open('naive_classifier.pickle','wb')
    pickle.dump(classifier,f)
    f.close()
    classifier.show_most_informative_features(10)
    accuracy = nltk.classify.accuracy(classifier,features[train_count:])
    print(accuracy.as_integer_ratio)

#train_classifier(features)
f = open('naive_classifier.pickle','rb')

classifier = pickle.load(f)
f.close()
features = extract_features("That movie is great movie")
accuracy = classifier.classify(features)
print("=================")
print(accuracy)
