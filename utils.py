from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import os


def vprint(*args, **kwargs):
    if os.getenv('VERBOSE'):
        print(*args, **kwargs)


def list_directory(directory):
    files = os.listdir(directory)
    return files


def tokenize_sentences(raw, language='english'):
    raw = raw.split('\n')
    raw_all = []
    for s in raw:
        raw_all.append(sent_tokenize(s, language=language))
    raw_all_flat = [item for sublist in raw_all for item in sublist]
    return raw_all_flat


def tokenize_word(raw, language='english'):
    return word_tokenize(raw, language=language)


def tokenize(raw):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(raw.lower())


def process_files(docindex):
    with open(docindex, 'r') as f:
        return [line.rstrip('\n') for line in f.readlines()]


def precision(TP, FP):
    return TP / (TP + FP)


def pprecision(tp, i):
    return tp / i


def recall(TP, FN):
    return TP / (TP + FN)


def F1_score(precision, recall):
    try:
        return 2 * (precision * recall / (precision + recall))
    except ZeroDivisionError:
        return 0


def baseline(doc, table, n, bm25=False):
    sentences = tokenize_sentences(doc, language='portuguese')
    return sentences[:5]


def summary(doc, table, n, bm25=False):
    dic = {}
    sims = table.similarity(doc, bm25=bm25)

    for i in range(table.n_docs()):
        dic[i] = sims[i][0]

    best = sorted(dic, key=dic.get, reverse=True)[:n]

    result = []
    corpus = table.get_original_corpus()
    for i in range(table.n_docs()):
        if i in best:
            result.append(corpus[i])
    return result


def MMR(doc, table, limit=5, k=0.3, remove_on_choice=False, bm25=True):
    ranks = {}  # Similarity rankings
    chosen = list()  # Chosen sentences set

    table.bm25vectorizer.use_idf = False
    while len(chosen) < limit:
        corpus = table.get_original_corpus()
        for i in range(len(corpus)):
            ranks[corpus[i]] = k * table.similarity(doc, bm25=bm25)[i]

            if len(chosen) > 0:
                ranks[corpus[i]] = ranks[corpus[i]] - \
                                   (1 - k) * sum([table.independent_similarity(v[0], corpus[i], bm25=bm25) for v in chosen])
            else:
                ranks[corpus[i]] = ranks[corpus[i]][0]

        choice = max(ranks, key=ranks.get)
        chosen.append(choice)
        ranks.pop(choice)
        if remove_on_choice:
            table.init([s for s in corpus if s != choice])  # Update table removing the chosen sentence
    table.bm25vectorizer.use_idf = True
    return chosen
