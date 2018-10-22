from collections   import defaultdict
from nltk.corpus   import stopwords
from nltk          import word_tokenize
from nltk.tokenize import RegexpTokenizer
from math          import log

import numpy as np

def tokenize(raw):
    tokenizer = RegexpTokenizer(r'\w+');
    return tokenizer.tokenize(raw.lower())

def process_files(docindex):
    with open(docindex, 'r') as f:
        return [line.rstrip('\n') for line in f.readlines()]

def qdict_weights(table, qtokens):
    qdict = defaultdict(int)

    for term in qtokens:
        qdict[term] += 1

    qdict_freqs = defaultdict(float)
    for term in qdict.keys():
        qdict_freqs[term] = qdict[term] / max(qdict.values())

    qdict_weights = []
    for term in qtokens:
        qdict_weights.append(qdict_freqs[term] * table.IDF(term))

    return qdict_weights

def similarity(table, query):
    qtokens = tokenize(query)
    qw = qdict_weights(table, qtokens)
    ws = table.weight_dict(qtokens)

    print("qv: " + str(qw))
    ranked_docs = {}
    for doc in ws.keys():
        print("dv[\"" + doc+ "\"]: " + str(ws[doc]))
        ranked_docs[doc] = np.array(qw).dot(np.array(ws[doc]))
    return ranked_docs

class table:
    def __init__(self, docindex = None):
        self.table  = defaultdict(lambda: defaultdict(int))
        self.itable = defaultdict(lambda: defaultdict(int))

        if docindex != None:
            self.init_from_file(docindex)

    def init_from_file(self, docindex):
        docnames = process_files(docindex)
        for docname in docnames:
            with open(docname, 'r') as f:
                for token in tokenize(f.read()):
                    self.itable[token][docname] += 1
                    self.table[docname][token] += 1

    def stats(self):
        return {'uniqterms': self.nouniqterms(),
                'termsno':   self.noterms(),
                'docsno':    self.nodocs()}

    def nodocs(self):
        return len(self.table.keys())

    def nouniqterms(self):
        return len(self.itable.keys())

    def noterms(self):
        no = 0
        for doc in self.table.keys():
            for term in self.table[doc].keys():
                no += self.table[doc][term]
        return no

    # Math
    def TF(self, doc, term):
        freq = self.table[doc][term]
        maximum = max(self.table[doc].values())
        return freq/maximum

    def DF(self, term):
        return len(self.itable[term])

    def IDF(self, term):
        try:
            #print("IDF of log({}/{})".format(self.nodocs(), self.DF(term)))
            return log(self.nodocs() / float (self.DF(term)), 2)
        except ZeroDivisionError:
            return 0

    def TF_IDF(self, doc, term):
        print("doc: {}\tterm: {}\tTF: {}\tIDF: {}".format(doc, term, self.TF(doc, term), self.IDF(term)))
        return self.TF(doc, term) * self.IDF(term)

    def weight_dict(self, order):
        ws = defaultdict(list)
        for doc in self.table.keys():
            for term in order:
                ws[doc].append(self.TF_IDF(doc, term))
        return ws
