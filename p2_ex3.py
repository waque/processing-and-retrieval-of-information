from eval import *
import os
import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd
from utils import tokenize_sentences, MMR
from table import Table
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.externals import joblib


def extract_features(doc, summary=None):
    tab = Table(parse_sentences=True, language='english')
    sentences = tokenize_sentences(doc)
    tab.init(sentences)

    tfidf = tab.similarity(doc, bm25=False)
    bm25 = tab.similarity(doc, bm25=True)

    svd = TruncatedSVD(n_components=11)
    svd.fit_transform(tab.matrix)

    if summary is not None:
        sum_sentences = tokenize_sentences(summary)

    sent_tfidf = []
    sent_bm25 = []
    sent_pos = []
    sent_len = []
    sent_label = []
    sent_svd = []

    tfidf = [val[0] for val in tfidf]
    bm25 = [val[0] for val in bm25]

    for i in range(len(sentences)):
        sent_tfidf.append(tfidf[i])
        sent_bm25.append(bm25[i])
        sent_pos.append(i+10)

        # sent_len.append(len(sentences[i]))
        words = tokenize(sentences[i])
        if len(words) == 0:
           avg_len = 0
        else:
            avg_len = sum([len(w) for w in words])/len(words)
        sent_len.append(avg_len)

        decomp = svd.transform(tab.tf_idf_vect([sentences[i]]))
        sent_svd.append(decomp[0][0])
        if summary is not None:
            if sentences[i] in sum_sentences:
                sent_label.append(1)
            else:
                sent_label.append(0)

    if summary is not None:
        return pd.DataFrame({'len': sent_len, 'svd': sent_svd, 'tfidf': sent_tfidf, 'bm25': sent_bm25}), pd.DataFrame({'label': sent_label})
    else:
        return pd.DataFrame({'len': sent_len, 'svd': sent_svd, 'tfidf': sent_tfidf, 'bm25': sent_bm25}), 0


def train_rfc(x, y):
    clf = RandomForestClassifier(n_estimators=50, max_depth=30)
    clf = clf.fit(x, y)
    return clf


def train_calibrated_perceptron(x, y):
    clf = Perceptron(penalty='l1', tol=1e-6)
    clf_iso = CalibratedClassifierCV(clf, cv=8, method='isotonic')
    clf_iso.fit(x, y)
    return clf_iso


def train_perceptron(x, y):
    clf = Perceptron()
    clf.fit(x, y)
    return clf


def train_mlp(x, y):
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, verbose=True, activation='identity', batch_size=100)
    clf.fit(x, y)
    return clf


def concat_dataset(x, y):
    return pd.concat(x, ignore_index=True), pd.concat(y, ignore_index=True)


def perceptron_summary(doc, n, clf):
    x, _ = extract_features(doc, None)
    y = clf.decision_function(x)

    sent = tokenize_sentences(doc)
    sorted_idx = np.argsort(-y).tolist()

    result = []
    for i in range(0, n):
        result.append(sent[sorted_idx.index(i)])

    return result


def classifier_summary(doc, n, clf):
    x, _ = extract_features(doc, None)
    y = clf.predict_proba(x)

    # Fetch probability of getting 1 (in summary)
    y = [v[1] for v in y]

    sentences = tokenize_sentences(doc)
    # sorted_idx = np.argsort(y).tolist()

    sentences_info = []
    for i in range(len(sentences)):
        sentences_info.append({'pos': i, 'text': sentences[i], 'y': y[i]})

    best = sorted(sentences_info, key=lambda k: k['y'], reverse=True)[:n]
    res = [s['text'] for s in best]
    return res


def main():
    x = []
    y = []
    train_directory = 'data/train/flat_text/'
    train_summary_directory = 'data/train/summary/'

    print("-------- Exercise 2 ---------")

    print("<<< Evaluating summaries using Multi-Layer Perceptron and set of features as described in report >>>")
    print('(Learning from files in directory ' + train_directory + ')')
    for filename in os.listdir(train_directory):
        with open(train_directory+filename, 'r', encoding='latin1') as file:
            d = file.read()
        with open(train_summary_directory+'Sum-'+filename, 'r', encoding='latin1') as file:
            s = file.read()

        x_, y_ = extract_features(d, s)
        x.append(x_)
        y.append(y_)

    x_train, y_train = concat_dataset(x, y)
    clf = train_mlp(x_train.as_matrix(), y_train.as_matrix().ravel())
    joblib.dump(clf, 'mlp.clf')

    evaluate_print('Classifier', *full_evaluate(classifier_summary, 5, clf))


if __name__ == '__main__':
    main()
