from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

from nltk.corpus import stopwords

def load_dataset():
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')

    return train, test

def vectorize(train, test):
    vect = TfidfVectorizer(use_idf=False)
    train_vec = vect.fit_transform(train.data)
    test_vec = vect.transform(test.data)
    return train_vec, test_vec

def count(train, test):
    counter = CountVectorizer()
    train_count = counter.fit(train.data)
    # No Oracle Processing
    #test_count = counter.fit(test.data)
    return counter

def filter(word_set, word):
    if word in word_set:
        return ''
    return word + ' '

def remove_stop_words(dataset):
    new = []
    stop = set(stopwords.words('english'))
    for entry in dataset.data:
        new.append("".join(filter(stop, word) for word in entry.split()))
    dataset.data = new

def filter_lt_p(vocab, word, p):
    if word in vocab and vocab[word] > p:
        return word + ' '
    return ''

def remove_rare_words(dataset, vocab, p):
    new = []
    vocab = vocab.copy()

    for entry in dataset.data:
        new.append("".join(filter_lt_p(vocab, word, p) for word in entry.split()))
    dataset.data = new

def train(classifier, train_vec, train_target):
    classifier.fit(train_vec, train_target)
    return classifier

def test(classifier, test_vec):
    pred = classifier.predict(test_vec)
    return pred

def evaluate(test_pred, test_target):
    print(metrics.accuracy_score(test_pred, test_target))
    print(metrics.classification_report(test_pred, test_target))

def cluster(train_vec, test_vec, n=20):
    cluster = MiniBatchKMeans(n)
    cluster.fit(train_vec)
    return cluster.fit(test_vec).labels_

def accum(vocab):
    total = 0
    for word, count in vocab.items():
        total = total + count
    return total


# Load dataset
train_data, test_data = load_dataset()

# Process dataset
remove_stop_words(train_data)
remove_stop_words(test_data)

freqs = count(train_data, test_data)

#Unnecessary, can be filtered during vectorization
#remove_rare_words(train_data, freqs.vocabulary_, 3)
#remove_rare_words(test_data, freqs.vocabulary_, 3)

train_vec, test_vec = vectorize(train_data, test_data)

# Train a classifier
classifier = MultinomialNB()
#classifier = KNeighborsClassifier()
#classifier = Perceptron()
#classifier = LinearSVC()
classifier = train(classifier, train_vec, train_data.target)

# Test and evaluate
pred = test(classifier, test_vec)
evaluate(pred, test_data.target)
